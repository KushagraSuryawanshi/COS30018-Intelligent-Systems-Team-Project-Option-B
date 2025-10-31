import os
from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.preprocessing.preprocessing import (
    to_grayscale,
    denoise,
    otsu_binarize,
    ensure_white_on_black,
    deskew,
    resize_and_center,
)

"""
Segmentation module for isolating and preparing digit images.

Functions:
    segment_image(...) -> Segments digits and returns boxes, crops, and overlay image.
    save_overlay(...) -> Writes overlay image to disk.
    save_crops(...) -> Saves cropped digit images (e.g., segmented_00.png, segmented_01.png).
    build_overlay_image(...) -> Builds a labeled preview overlay with bounding boxes.

CLI Usage:
    python -m src.segmentation.segment path/to/image.png
"""

DEFAULT_OUT_DIR = "outputs"
DEFAULT_BIN_DIR = "binarized"


def _binarize_image(
        img_gray: np.ndarray,
        save_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Otsu thresholding and morphological cleaning to obtain a binary mask.

    :param img_gray: Grayscale input image.
    :param save_dir: Optional directory to save binarized results.
    :return: Tuple of (binary_inv, cleaned) as uint8 arrays.
    """
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, th_inv = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "binarized.png"), th_inv)
        cv2.imwrite(os.path.join(save_dir, "binarized_clean.png"), cleaned)

    return th_inv, cleaned


def find_digit_contours_from_binary(
        binary_img: np.ndarray,
        min_area: int = 50
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes for connected components in a binary image.

    :param binary_img: Binary image (white foreground on black background).
    :param min_area: Minimum bounding box area to keep.
    :return: List of bounding boxes sorted from left to right.
    """
    binu = binary_img.copy()
    if binu.dtype != np.uint8:
        binu = (np.clip(binu, 0, 1) * 255).astype("uint8")

    contours, _ = cv2.findContours(
        binu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    height, width = binu.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        if w >= 0.9 * width and h >= 0.9 * height:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes


def crop_from_boxes(
        gray_image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        pad: int = 4
) -> List[np.ndarray]:
    """
    Crop a grayscale image using bounding boxes with optional padding.

    :param gray_image: Input grayscale image.
    :param boxes: List of bounding boxes (x, y, w, h).
    :param pad: Padding around each crop.
    :return: List of cropped grayscale images.
    """
    height, width = gray_image.shape[:2]
    crops = []

    for x, y, w, h in boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        crop = gray_image[y0:y1, x0:x1]
        crops.append(crop)

    return crops


def save_crops(
        crops: List[np.ndarray],
        out_dir: str = DEFAULT_OUT_DIR,
        prefix: str = "segmented"
):
    """
    Save cropped grayscale digit images to disk.

    :param crops: List of image crops to save.
    :param out_dir: Output directory for saved crops.
    :param prefix: File prefix (e.g., 'segmented').
    :return: List of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for i, crop in enumerate(crops):
        fname = f"{prefix}_{i: 02d}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, crop)
        saved_paths.append(path)

    return saved_paths


def build_overlay_image(
        color_image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        labels: Optional[List[str]] = None,
        pad: int = 0
):
    """
    Draw bounding boxes (and optional labels) on a color image.

    :param color_image: Input BGR or grayscale image.
    :param boxes: List of bounding boxes (x, y, w, h).
    :param labels: Optional list of labels corresponding to boxes.
    :param pad: Extra padding for drawn rectangles.
    :return: Annotated image with boxes and optional labels.
    """
    vis = color_image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(
            vis, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2
        )
        if labels is not None:
            label = str(labels[i]) if i < len(labels) else str(i)
            cv2.putText(
                vis,
                label,
                (x, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
    return vis


def segment_image(
        image_path: str,
        out_dir: str = DEFAULT_OUT_DIR,
        bin_dir: str = DEFAULT_BIN_DIR,
        pad: int = 4,
        min_area: int = 50,
        save_crops_flag: bool = True,
        return_overlay: bool = True,
):
    """
    Full segmentation pipeline for digit extraction.

    :param image_path: Path to input image.
    :param out_dir: Directory for cropped output images.
    :param bin_dir: Directory for saving binarized intermediates.
    :param pad: Padding around digit bounding boxes.
    :param min_area: Minimum box area to retain.
    :param save_crops_flag: Whether to save cropped digit images.
    :param return_overlay: Whether to return overlay image in result.
    :return: Dictionary containing boxes, crops, overlay, and saved paths.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 1: Binarize and save results
    _, cleaned = _binarize_image(gray, save_dir=bin_dir)

    # Step 2: Detect bounding boxes
    boxes = find_digit_contours_from_binary(cleaned, min_area=min_area)

    # Step 3: Crop digits from grayscale image
    crops = crop_from_boxes(gray, boxes, pad=pad)

    # Step 4: Deskew, center, and resize to 28x28
    centered_crops = []
    for crop in crops:
        bin_crop = ensure_white_on_black(crop)
        deskewed = deskew(bin_crop)
        centered = resize_and_center(deskewed, size=(28, 28), pad=4)
        centered_crops.append(centered)

    # Step 5: Optionally save crops
    saved_paths = []
    if save_crops_flag:
        saved_paths = save_crops(crops, out_dir=out_dir, prefix="segmented")

    # Step 6: Build overlay visualization
    color_src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_src is None:
        color_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = build_overlay_image(color_src, boxes, labels=None, pad=pad)

    # Step 7: Save labeled preview overlay
    preview_labels = [str(i) for i in range(len(boxes))]
    preview_path = os.path.join(out_dir, "preview_labeled.png")
    cv2.imwrite(preview_path, build_overlay_image(color_src, boxes, labels=preview_labels, pad=pad))

    result = {
        "image_path": image_path,
        "boxes": boxes,
        "crops": crops,
        "centered_crops": centered_crops,
        "saved_paths": saved_paths,
        "overlay": overlay if return_overlay else None,
        "binary": cleaned,
    }

    return result


def _cli():
    """Command-line interface for quick segmentation testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Segment digits from an input image.")
    parser.add_argument("image", help="Input image path (grayscale or color).")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory for crops/preview.")
    parser.add_argument("--bin", default=DEFAULT_BIN_DIR, help="Binarized outputs directory.")
    parser.add_argument("--pad", type=int, default=4, help="Padding around boxes when cropping.")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum bounding box area to keep.")
    args = parser.parse_args()

    res = segment_image(args.image, out_dir=args.out, bin_dir=args.bin, pad=args.pad, min_area=args.min_area)
    print(f"[Segment] Found {len(res['boxes'])} boxes. Saved {len(res['saved_paths'])} crops to {args.out}")
    print("Preview written to:", os.path.join(args.out, "preview_labeled.png"))


if __name__ == "__main__":
    _cli()