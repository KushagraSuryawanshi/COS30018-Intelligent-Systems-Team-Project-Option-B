import os
import uuid
import threading
import torch
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify
)
import cv2

from src.segmentation.segment import segment_image, build_overlay_image
from src.preprocessing.preprocessing import (
    prepare_for_fnn, prepare_for_seq,
    prepare_for_cnn, prepare_for_emnist
)
from src.models.FNN.fnn_wrapper import FNNModel
from src.models.SEQ.seq_wrapper import SeqModel
from src.models.CNN.cnn_wrapper import CNNModel
from src.models.CNN_EMNIST.cnn_emnist_wrapper import EMNISTModel
from src.models.CNN.CNN_train import train_main as train_cnn
from src.models.FNN.FNN import train_and_save as train_fnn
from src.models.SEQ.seq_train import train_and_save as train_seq
from src.models.CNN_EMNIST.cnn_emnist_train import train_main as train_emnist


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Define key directories for project and data handling
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(GUI_DIR, "..", ".."))

STATIC_DIR = os.path.join(GUI_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
OVERLAY_DIR = os.path.join(STATIC_DIR, "overlays")
MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "models", "saved_models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model file paths
FNN_MODEL_PATH = os.path.join(MODELS_DIR, "fnn_net.pt")
SEQ_MODEL_PATH = os.path.join(MODELS_DIR, "Sequential.keras")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model_best.pth")
EMNIST_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_emnist_byclass.pth")

# Select device for PyTorch (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Application is running on device: {DEVICE}")

# Cached models in memory
_models = {"fnn": None, "seq": None, "cnn": None, "emnist": None}

# Default training settings for each model type
SETTINGS = {
    "model_choice": "ensemble",
    "cnn": {"epochs": 12, "learning_rate": 0.001, "batch_size": 128},
    "seq": {"epochs": 5, "learning_rate": 0.001, "batch_size": 128},
    "fnn": {"epochs": 10, "learning_rate": 0.05, "batch_size": 200},
    "emnist": {"epochs": 10, "learning_rate": 0.001, "batch_size": 64},
}

# Track background training status
TRAINING_STATUS = {"status": "idle", "message": ""}


def load_models():
    """
    Load all trained machine learning models from disk into memory.

    :return: None
    """
    if _models["fnn"] is None and os.path.exists(FNN_MODEL_PATH):
        _models["fnn"] = FNNModel(FNN_MODEL_PATH, device=DEVICE)
    if _models["seq"] is None and os.path.exists(SEQ_MODEL_PATH):
        _models["seq"] = SeqModel(SEQ_MODEL_PATH)
    if _models["cnn"] is None and os.path.exists(CNN_MODEL_PATH):
        _models["cnn"] = CNNModel(CNN_MODEL_PATH, device=DEVICE)
    if _models["emnist"] is None and os.path.exists(EMNIST_MODEL_PATH):
        try:
            _models["emnist"] = EMNISTModel(EMNIST_MODEL_PATH, device=DEVICE)
            app.logger.info(f"Loaded EMNIST model from {EMNIST_MODEL_PATH}")
        except Exception as e:
            app.logger.error(f"Failed to load EMNIST model: {e}")

    if not any(_models.values()):
        app.logger.error("No models loaded. Predictions will not function.")


def run_training_task(model_to_train, params):
    """
    Run model training in a background thread.

    :param model_to_train: Model type ('cnn', 'fnn', 'seq')
    :param params: Dictionary containing training parameters (epochs, lr, batch_size)
    :return: None
    """
    global TRAINING_STATUS
    try:
        TRAINING_STATUS["status"] = "running"
        TRAINING_STATUS["message"] = f"Training {model_to_train.upper()}..."
        app.logger.info(
            f"Starting training for {model_to_train.upper()} with params: {params}"
        )

        # Map each model type to its corresponding training function and parameter keys
        training_map = {
            "cnn": (train_cnn, {"epochs": "epochs", "lr": "lr",
                                "batch_size": "batch_size"}),
            "fnn": (train_fnn, {"epochs": "epochs", "lr": "lr",
                                "batch_size": "batch_size"}),
            "seq": (train_seq, {"epochs": "epochs",
                                "batch_size": "batch_size"}),
            "emnist": (train_emnist, {"epochs": "epochs", "lr": "lr",
                                     "batch_size": "batch_size"}),
        }



        if model_to_train in training_map:
            train_func, param_keys = training_map[model_to_train]
            kwargs = {key: params[val] for key, val in param_keys.items()}
            train_func(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_to_train}")

        # Reload the trained model into memory after completion
        _models[model_to_train] = None
        load_models()

        TRAINING_STATUS["status"] = "completed"
        TRAINING_STATUS["message"] = (
            f"{model_to_train.upper()} model training completed successfully."
        )
        app.logger.info(TRAINING_STATUS["message"])

    except Exception as e:
        app.logger.error(
            f"Training failed for {model_to_train}: {e}", exc_info=True
        )
        TRAINING_STATUS["status"] = "error"
        TRAINING_STATUS["message"] = f"An error occurred: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the main page logic, including image uploads and model predictions.

    :return: Rendered HTML page with prediction results and overlay image.
    """
    overlay_url = None
    predictions = []

    if request.method == "POST":
        # Validate uploaded file
        if "image" not in request.files:
            flash("No file part in request", "error")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        # Save uploaded image to local storage
        uid = uuid.uuid4().hex[:8]
        upload_name = f"upload_{uid}{os.path.splitext(file.filename)[1]}"
        upload_path = os.path.join(UPLOAD_DIR, upload_name)
        file.save(upload_path)

        # Load all models into memory if not already loaded
        load_models()
        if not any(_models.values()):
            flash("No models found. Check saved_models directory.", "error")
            return redirect(request.url)

        SETTINGS["model_choice"] = request.form.get("model_choice", "ensemble")

        # Perform segmentation on uploaded image
        seg_res = segment_image(upload_path, min_area=30)

        # Determine which models to run
        run_cnn = SETTINGS["model_choice"] in ("ensemble", "cnn")
        run_seq = SETTINGS["model_choice"] in ("ensemble", "seq")
        run_fnn = SETTINGS["model_choice"] in ("ensemble", "fnn")
        run_emnist = SETTINGS["model_choice"] in ("ensemble", "emnist")

        # Predict for each segmented character region
        for centered in seg_res["centered_crops"]:
            crop_pred = {}

            if run_fnn and _models.get("fnn"):
                fnn_in = prepare_for_fnn(centered)
                lab, conf = _models["fnn"].predict_from_preprocessed(fnn_in)
                crop_pred["fnn"] = {"label": lab, "conf": conf}

            if run_seq and _models.get("seq"):
                seq_in = prepare_for_seq(centered)
                lab, conf = _models["seq"].predict_from_preprocessed(seq_in)
                crop_pred["seq"] = {"label": lab, "conf": conf}

            if run_cnn and _models.get("cnn"):
                cnn_in = prepare_for_cnn(centered)
                lab, conf = _models["cnn"].predict_from_preprocessed(cnn_in)
                crop_pred["cnn"] = {"label": lab, "conf": conf}

            if run_emnist and _models.get("emnist"):
                emnist_in = prepare_for_emnist(centered, device=DEVICE)
                lab, conf = _models["emnist"].predict_from_preprocessed(emnist_in)
                crop_pred["emnist"] = {"label": lab, "conf": conf}

            predictions.append(crop_pred)

        # Build overlay image showing detected and labeled characters
        color_img = cv2.imread(upload_path, cv2.IMREAD_COLOR)
        overlay_name = f"overlay_{uid}.png"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_name)

        # Choose one predicted label per segment for overlay
        overlay_labels = []
        for p in predictions:
            chosen = p.get("emnist") or p.get("cnn") or p.get("seq") or p.get("fnn")
            overlay_labels.append(
                chosen["label"] if chosen and chosen["label"] is not None else ""
            )

        overlay_img = build_overlay_image(
            color_img, seg_res["boxes"], labels=overlay_labels
        )
        cv2.imwrite(overlay_path, overlay_img)
        overlay_url = url_for("static", filename=f"overlays/{overlay_name}")

    return render_template(
        "index.html",
        overlay_url=overlay_url,
        predictions=predictions,
        settings=SETTINGS
    )


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Start a background training job for the selected model.

    :return: JSON response indicating job status.
    """
    global TRAINING_STATUS
    if TRAINING_STATUS["status"] == "running":
        return jsonify({
            "status": "error",
            "message": "A training job is already in progress."
        }), 409

    try:
        model_to_train = request.form.get("model_to_train")
        if not model_to_train:
            return jsonify({
                "status": "error",
                "message": "No model selected."
            }), 400

        params = {
            "epochs": int(request.form.get(f"{model_to_train}_epochs")),
            "lr": float(request.form.get(f"{model_to_train}_lr")),
            "batch_size": int(request.form.get(f"{model_to_train}_bs")),
        }

        # Launch training in a separate thread to keep the app responsive
        train_thread = threading.Thread(
            target=run_training_task, args=(model_to_train, params)
        )
        train_thread.start()

        return jsonify({"status": "success", "message": "Training started."})

    except (ValueError, KeyError) as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid parameter: {e}"
        }), 400


@app.route("/retrain-status", methods=["GET"])
def retrain_status():
    """
    Return the current training job status.

    :return: JSON with training status and progress message.
    """
    return jsonify(TRAINING_STATUS)


if __name__ == "__main__":
    # Used to enable/disable debugging mode
    app.run(debug=False, host="0.0.0.0", port=8080)