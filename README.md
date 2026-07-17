# Penerapan Transfer Learning VGG16 untuk Klasifikasi Jenis Alat Musik Tradisional Indonesia Berbasis Web Menggunakan Flask

Aplikasi web untuk mengklasifikasikan jenis alat musik tradisional Indonesia
dari gambar yang diunggah, menggunakan model **Transfer Learning VGG16**
dan **Flask** sebagai backend web.

## Struktur Project

```
alat_musik_klasifikasi/
├── app.py                  # Aplikasi Flask utama
├── train_model.py          # Script untuk melatih model VGG16
├── requirements.txt        # Daftar dependency Python
├── dataset/
│   ├── train/               # Dataset training (per folder kelas)
│   └── validation/           # Dataset validasi (per folder kelas)
├── model/
│   ├── vgg16_alat_musik.h5   # Model hasil training (dihasilkan otomatis)
│   └── class_indices.json    # Mapping label kelas (dihasilkan otomatis)
├── static/
│   ├── css/style.css        # Styling halaman
│   └── uploads/             # Tempat menyimpan gambar yang diunggah user
└── templates/
    └── index.html           # Halaman utama (form upload + hasil prediksi)
```

## Kelas Alat Musik (Dataset Terpasang)

Dataset yang sudah disertakan di folder `dataset/` bersumber dari Roboflow
Universe (lisensi CC BY 4.0) dan berisi **12 kelas** alat musik tradisional
berdasarkan daerah asal & cara memainkannya:

| No | Nama Kelas (folder)     | Alat Musik  | Daerah Asal   | Cara Main |
|----|--------------------------|-------------|----------------|-----------|
| 1  | `bali_suling`            | Suling      | Bali           | Tiup      |
| 2  | `batak_suling`           | Suling      | Batak          | Tiup      |
| 3  | `betawi_tehyan`          | Tehyan      | Betawi         | Gesek     |
| 4  | `betawi_trombone`        | Trombone    | Betawi         | Tiup      |
| 5  | `maluku_tahuri`          | Tahuri      | Maluku         | Tiup      |
| 6  | `minangkabau_sarunai`    | Sarunai     | Minangkabau    | Tiup      |
| 7  | `papua_triton`           | Triton      | Papua          | Tiup      |
| 8  | `sunda_rebab`            | Rebab       | Sunda          | Gesek     |
| 9  | `sunda_tarawangsa`       | Tarawangsa  | Sunda          | Gesek     |
| 10 | `sunda_suling`           | Suling      | Sunda          | Tiup      |
| 11 | `sunda_taleot`           | Taleot      | Sunda          | Tiup      |
| 12 | `sunda_terompet`         | Terompet    | Sunda          | Tiup      |

**Jumlah gambar:**
- Training: 2.550 gambar (folder `dataset/train/`)
- Validasi: 215 gambar, gabungan split `valid` + `test` dari sumber asli
  (folder `dataset/validation/`)

> Dataset ini sudah siap pakai — tidak perlu mengumpulkan gambar tambahan
> untuk mulai training. Sumber: Roboflow Universe,
> https://universe.roboflow.com/dataset-nr5wu/alat-musik-slwde

## Cara Menjalankan

### 1. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 2. Install dependency
```bash
pip install -r requirements.txt
```

### 3. Dataset
Dataset **sudah tersedia** di folder `dataset/train/` (2.550 gambar) dan
`dataset/validation/` (215 gambar), terbagi ke 12 folder kelas seperti
dijelaskan di atas. Anda tidak perlu mengumpulkan gambar tambahan untuk
mulai training — langsung lanjut ke langkah berikutnya.

Jika ingin menambah data sendiri, cukup masukkan gambar baru ke folder
kelas yang sesuai (atau buat folder kelas baru) mengikuti pola:
```
dataset/train/<nama_kelas>/gambar1.jpg
dataset/validation/<nama_kelas>/gambar1.jpg
```

### 4. Latih model
```bash
python train_model.py
```
Model terbaik akan otomatis tersimpan di `model/vgg16_alat_musik.h5`,
beserta `model/class_indices.json` yang menyimpan urutan label kelas.

### 5. Jalankan aplikasi web
```bash
python app.py
```
Buka browser ke `http://127.0.0.1:5000`

## Metode Transfer Learning

Project ini menggunakan pendekatan **Feature Extraction**:
- Layer convolutional VGG16 (pretrained ImageNet) dibekukan (`trainable = False`)
- Ditambahkan layer `Flatten → Dense(256, relu) → Dropout(0.5) → Dense(jumlah_kelas, softmax)`
- Hanya layer baru ini yang dilatih ulang menggunakan dataset alat musik

Pendekatan ini cocok untuk dataset yang berjumlah relatif kecil karena
memanfaatkan fitur visual umum yang sudah dipelajari VGG16 dari 14 juta
gambar ImageNet.

## Catatan

- Format gambar yang didukung: JPG, JPEG, PNG (maksimal 5MB)
- Ukuran input gambar otomatis di-resize ke 224x224 piksel (standar VGG16)
- Jika ingin meningkatkan akurasi, bisa dicoba pendekatan **Fine-Tuning**
  dengan membuka beberapa layer terakhir VGG16 untuk dilatih ulang bersama
  dataset baru.
