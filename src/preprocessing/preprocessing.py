import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

"""
Preprocessing utilities for segmentation and model input preparation.

Functions:
 - read_image(path, color=True)
 - to_grayscale(img)
 - denoise(img, method='gaussian')
 - otsu_binarize(img)
 - ensure_white_on_black(img)
 - deskew(img)
 - resize_and_center(img, size=(28, 28))
 - prepare_for_fnn(img)
 - prepare_for_cnn(img)
 - prepare_for_seq(img)
 - prepare_for_emnist(img)
"""

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def read_image(path: str, color=True):
    """
    Read an image from disk.

    :param path: Path to the image file.
    :param color: If True, reads in color mode. Otherwise grayscale.
    :return: Loaded image as NumPy array.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_grayscale(img: np.ndarray):
    """
    Convert a BGR image to grayscale if necessary.

    :param img: Input image.
    :return: Grayscale image.
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def denoise(img: np.ndarray, method='gaussian'):
    """
    Apply simple denoising (Gaussian or median blur).

    :param img: Input image.
    :param method: Denoising method ('gaussian' or 'median').
    :return: Denoised image.
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (3, 3), 0)
    if method == 'median':
        return cv2.medianBlur(img, 3)
    return img


def otsu_binarize(gray_img: np.ndarray):
    """
    Apply Otsu's thresholding with inversion.

    :param gray_img: Input grayscale image.
    :return: Binary image (uint8).
    """
    if gray_img.dtype != np.uint8:
        gray = (np.clip(gray_img, 0, 1) * 255).astype('uint8')
    else:
        gray = gray_img
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def enhance_for_models(img: np.ndarray, use_clahe=True, dilate=True):
    """
    Apply preprocessing enhancements to bring custom digits closer to MNIST style.

    :param img: uint8 grayscale (28x28), digits white on black background.
    :param use_clahe: Whether to apply CLAHE contrast enhancement.
    :param dilate: Whether to apply dilation and slight blur.
    :return: Enhanced image.
    """
    out = img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if dilate:
        th = cv2.threshold(out, 10, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        out = cv2.dilate(th, kernel, iterations=1)
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


def ensure_white_on_black(img: np.ndarray):
    """
    Ensure image polarity is white digits on black background.

    - Compute both normal and inverted Otsu binarizations.
    - Choose the version with smaller foreground area (digits are small).
    - Return a cleaned binary image with digits white on black.

    :param img: Input image.
    :return: Binary image with white foreground on black.
    """
    gray = to_grayscale(img)
    if gray.dtype != np.uint8:
        gray8 = (np.clip(gray, 0, 1) * 255).astype('uint8')
    else:
        gray8 = gray.copy()

    # Compute Otsu binarization (normal and inverted)
    _, th_normal = cv2.threshold(
        gray8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(
        gray8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Foreground area counts
    area_normal = int(np.count_nonzero(th_normal))
    area_inv = int(np.count_nonzero(th_inv))

    # Choose the smaller foreground (digits are small)
    chosen = th_inv if area_inv <= area_normal else 255 - th_normal

    # Apply small morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    chosen_clean = cv2.morphologyEx(chosen, cv2.MORPH_OPEN, kernel, iterations=1)
    return chosen_clean


def deskew(img: np.ndarray):
    """
    Deskew a grayscale image using its image moments.

    :param img: Input grayscale image.
    :return: Deskewed image.
    """
    gray = img if img.dtype == np.uint8 else (img * 255).astype('uint8')
    moments = cv2.moments(gray)
    if abs(moments["mu02"]) < 1e-2:
        return gray

    skew = moments["mu11"] / moments["mu02"]
    height, width = gray.shape[:2]
    transform_matrix = np.float32([
        [1, skew, -0.5 * width * skew],
        [0, 1, 0]
    ])
    return cv2.warpAffine(
        gray,
        transform_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def resize_and_center(img, size=(28, 28), pad=4):
    """
    Resize keeping aspect ratio and center into a target canvas.

    :param img: Grayscale NumPy array (H, W), white foreground on black.
    :param size: Target output size (width, height).
    :param pad: Padding to leave around the digit.
    :return: Resized and centered image.
    """
    height, width = img.shape[:2]

    # Find bounding box of nonzero pixels
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = img[y0:y1 + 1, x0:x1 + 1]

    # Compute scaling to fit within target - padding
    target_w, target_h = size
    max_w = target_w - pad * 2
    max_h = target_h - pad * 2
    ch, cw = crop.shape[:2]
    scale = min(max_w / cw, max_h / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center resized digit on black background
    canvas = np.zeros(size, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _prepare_image_array(crop_image):
    """
    Shared preprocessing step to convert any image to normalized grayscale array.

    :param crop_image: PIL Image or NumPy array.
    :return: Normalized grayscale NumPy array (float32, values 0–1).
    """
    if not isinstance(crop_image, Image.Image):
        crop_image = Image.fromarray(crop_image)

    img = crop_image.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0

    # Ensure polarity is white-on-black
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def prepare_for_fnn(crop_image):
    """
    Preprocess for FNN input.

    :param crop_image: PIL Image or NumPy array (H, W) or color image.
    :return: NumPy array shape (1, 784), normalized using MNIST stats.
    """
    arr = _prepare_image_array(crop_image)
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr.reshape(1, -1).astype('float32')


def prepare_for_cnn(crop_image, device='cpu'):
    """
    Preprocess for CNN input.

    :param crop_image: PIL Image or NumPy array.
    :param device: Target device ('cpu' or 'cuda').
    :return: Torch tensor of shape (1, 1, 28, 28).
    """
    arr = _prepare_image_array(crop_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    img2 = Image.fromarray((arr * 255).astype('uint8'))
    tensor = transform(img2).unsqueeze(0)
    return tensor.to(device)


def prepare_for_seq(crop_image):
    """
    Preprocess for sequential model input.

    :param crop_image: PIL Image or NumPy array.
    :return: NumPy array of shape (1, 28, 28), normalized similar to Keras usage.
    """
    import tensorflow as tf

    arr = _prepare_image_array(crop_image)
    arr = tf.keras.utils.normalize(arr, axis=1)
    return arr.reshape(1, 28, 28).astype('float32')


def prepare_for_emnist(crop_image, device='cpu'):
    """
    Preprocess image for EMNIST ByClass CNN+ model.

    :param crop_image: PIL Image or NumPy array.
    :param device: Torch device string ('cpu' or 'cuda').
    :return: Torch tensor of shape (1, 1, 28, 28), normalized using (0.5, 0.5).
    """
    arr = _prepare_image_array(crop_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img2 = Image.fromarray((arr * 255).astype('uint8'))
    tensor = transform(img2).unsqueeze(0)
    return tensor.to(device)