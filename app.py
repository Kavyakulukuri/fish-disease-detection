from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "C:/Users/kavya/Desktop/Fish disease/fish_disease_vgg16_model.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model = None  # Avoid crashing if the model is missing

# Define class labels
class_labels = {
    0: "Healthy Fish",
    1: "Bacterial Infection",
    2: "Fungal Infection",
    3: "Parasitic Infection",
    4: "Viral Infection"
}

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Creates folder if it doesn’t exist

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_fish_disease(img_path):
    """Predict fish disease from an image."""
    try:
        if model is None:
            return "Model not loaded properly. Check the model file path."

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_labels.get(predicted_class_index, "Unknown Disease")

        return predicted_class_name
    except Exception as e:
        return str(e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        prediction = predict_fish_disease(file_path)
        image_url = f"http://127.0.0.1:5000/uploads/{filename}"
        return jsonify({"prediction": prediction, "image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve the uploaded image."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Runs on all network interfaces
