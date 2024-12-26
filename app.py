import os
import cv2
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from flask import Flask, request, jsonify
import torch
from sklearn.metrics.pairwise import euclidean_distances

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Access environment variables
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")

# Use the loaded environment variables in your app
app.config["SECRET_KEY"] = FLASK_SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 1. Kaggle Dataset Setup
def download_kaggle_dataset():
    """Download the Kaggle dataset."""
    if not os.path.exists("fashion-dataset"):
        os.makedirs("fashion-dataset")
    os.system(
        "kaggle datasets download -d paramaggarwal/fashion-product-images-dataset -p fashion-dataset --unzip"
    )

# Ensure dataset is available
download_kaggle_dataset()

# Path to dataset images
DATASET_PATH = "fashion-dataset/fashion-dataset/images"

# 2. Feature Extraction Model Setup
def build_feature_extraction_model():
    """Build the ResNet50 feature extraction model."""
    base_model = ResNet50(weights="imagenet", include_top=False)
    base_model.trainable = False
    pooling_layer = GlobalMaxPool2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=pooling_layer)
    return model

# Initialize the model
model = build_feature_extraction_model()

# 3. Extract Features from Images
def extract_features(image_path):
    """Extract features from a single image."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    return result / norm(result)

# Extract features from the entire dataset
def extract_features_for_dataset(dataset_path):
    """Extract features for all images in the dataset."""
    filenames = [
        os.path.join(dataset_path, file)
        for file in os.listdir(dataset_path)
        if file.endswith((".jpg", ".jpeg", ".png"))
    ]
    features = [extract_features(file) for file in filenames]
    return filenames, features

filenames, image_features = extract_features_for_dataset(DATASET_PATH)

# Save features and filenames for future use
with open("filenames.pkl", "wb") as f:
    pkl.dump(filenames, f)

with open("image_features.pkl", "wb") as f:
    pkl.dump(image_features, f)

# 4. Load Pickled Data (if already extracted)
def load_pickled_data():
    """Load pickled filenames and features."""
    with open("filenames.pkl", "rb") as f:
        filenames = pkl.load(f)
    with open("image_features.pkl", "rb") as f:
        image_features = pkl.load(f)
    return filenames, image_features

# 5. Nearest Neighbor Model for Similarity
neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
neighbors.fit(image_features)

# 6. Find Similar Images
def find_similar_images(input_image_path):
    """Find similar images for a given input image."""
    input_feature = extract_features(input_image_path)
    distances, indices = neighbors.kneighbors([input_feature])
    similar_images = [filenames[i] for i in indices[0]]
    return similar_images

# 7. Dominant Color Extraction
def find_dominant_color(image, clusters=3, max_iter=100):
    """Find dominant colors in an image using clustering."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]
    if len(pixels) == 0:
        return np.array([])

    pixels_tensor = torch.tensor(pixels, dtype=torch.float32)
    if torch.cuda.is_available():
        pixels_tensor = pixels_tensor.cuda()

    centroids = pixels_tensor[torch.randperm(len(pixels_tensor))[:clusters]]

    for _ in range(max_iter):
        distances = torch.cdist(pixels_tensor, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([pixels_tensor[labels == i].mean(0) for i in range(clusters)])
        if torch.allclose(centroids, new_centroids, atol=1e-2):
            break
        centroids = new_centroids

    return centroids.cpu().numpy().astype(int)

# 8. Flask API Endpoints

@app.route("/dominant-colors", methods=["POST"])
def dominant_colors():
    """API to get dominant colors of an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = f"./uploaded_images/{image_file.filename}"
    os.makedirs("./uploaded_images", exist_ok=True)
    image_file.save(image_path)

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    dominant_colors = find_dominant_color(image)
    os.remove(image_path)

    return jsonify({"dominant_colors": dominant_colors.tolist()})

@app.route("/find-similar", methods=["POST"])
def find_similar():
    """API to find similar images for an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = f"./uploaded_images/{image_file.filename}"
    os.makedirs("./uploaded_images", exist_ok=True)
    image_file.save(image_path)

    similar_images = find_similar_images(image_path)
    os.remove(image_path)

    return jsonify({"similar_images": similar_images})

# Start the Flask app
if __name__ == "__main__":
    # Bind to the port provided by Render
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not provided
    app.run(host="0.0.0.0", port=port)
