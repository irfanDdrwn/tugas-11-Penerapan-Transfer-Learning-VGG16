"""
Aplikasi Web Klasifikasi Jenis Alat Musik Tradisional Indonesia
Menggunakan Transfer Learning VGG16 + Flask

Cara pakai:
1. Latih model dulu dengan train_model.py (menghasilkan model/vgg16_alat_musik.h5)
2. Jalankan aplikasi ini: python app.py
3. Buka browser ke http://127.0.0.1:5000

Untuk deploy ke platform seperti Railway/Render (di mana file model besar
tidak bisa ikut ter-push ke GitHub karena limit 100MB), set environment
variable MODEL_URL ke link download langsung file .h5 (misal dari GitHub
Releases), dan aplikasi akan otomatis download model itu saat pertama kali start.
"""

import os

# Batasi TensorFlow supaya tidak alokasikan memori berlebihan (penting
# untuk platform deploy dengan RAM terbatas seperti Railway)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import requests
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# ==============================
# KONFIGURASI
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'vgg16_alat_musik.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

# URL download model, di-set lewat environment variable MODEL_URL
# (misalnya link file dari GitHub Releases). Kosongkan/biarkan default
# jika model sudah ada langsung di folder model/ (misal saat jalan di lokal).
MODEL_URL = os.environ.get('MODEL_URL', '')

# Urutan kelas HARUS sama persis dengan urutan folder saat training
# (disimpan otomatis oleh train_model.py ke class_indices.json)
import json
CLASS_INDEX_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # maksimal 5 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODEL & LABEL KELAS
# ==============================
model = None
class_labels = []

def download_model_if_needed():
    """Download file model dari MODEL_URL jika belum ada di lokal.
    Berguna untuk platform deploy (Railway/Render) di mana file besar
    tidak ikut ter-push lewat GitHub."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        print("[PERINGATAN] MODEL_URL tidak di-set dan file model tidak ditemukan lokal.")
        return

    print(f"[INFO] Model tidak ditemukan lokal, mendownload dari {MODEL_URL} ...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Model berhasil didownload ke {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Gagal download model: {e}")


def load_trained_model():
    """Load model dan label kelas jika file model tersedia."""
    global model, class_labels
    download_model_if_needed()
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        if os.path.exists(CLASS_INDEX_PATH):
            with open(CLASS_INDEX_PATH, 'r') as f:
                class_indices = json.load(f)
            # class_indices: {"angklung": 0, "gamelan": 1, ...} -> balik jadi list terurut index
            class_labels = [None] * len(class_indices)
            for label, idx in class_indices.items():
                class_labels[idx] = label
        print(f"[INFO] Model berhasil dimuat dari {MODEL_PATH}")
        print(f"[INFO] Kelas: {class_labels}")
    else:
        print(f"[PERINGATAN] Model belum ditemukan di {MODEL_PATH}.")
        print("[PERINGATAN] Jalankan train_model.py terlebih dahulu untuk melatih model.")

load_trained_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    """Melakukan prediksi kelas alat musik dari sebuah gambar."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]
    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index]) * 100

    label = class_labels[top_index] if class_labels else f"Kelas {top_index}"

    # Ambil 3 prediksi teratas untuk ditampilkan sebagai info tambahan
    top_indices = predictions.argsort()[-3:][::-1]
    top_predictions = [
        {
            "label": class_labels[i] if class_labels else f"Kelas {i}",
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top_indices
    ]

    return label, round(confidence, 2), top_predictions


# ==============================
# ROUTES
# ==============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_ready=model is not None)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template(
            'index.html',
            model_ready=False,
            error="Model belum tersedia. Jalankan train_model.py terlebih dahulu."
        )

    if 'file' not in request.files:
        return render_template('index.html', model_ready=True, error="Tidak ada file yang diunggah.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', model_ready=True, error="Silakan pilih gambar terlebih dahulu.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        label, confidence, top_predictions = predict_image(filepath)

        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template(
            'index.html',
            model_ready=True,
            prediction=label,
            confidence=confidence,
            top_predictions=top_predictions,
            image_path=image_url
        )
    else:
        return render_template(
            'index.html',
            model_ready=True,
            error="Format file tidak didukung. Gunakan JPG, JPEG, atau PNG."
        )


if __name__ == '__main__':
    app.run(debug=True)
