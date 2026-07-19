"""
Script Training Model - Transfer Learning VGG16
Klasifikasi Jenis Alat Musik Tradisional Indonesia

STRUKTUR DATASET YANG DIBUTUHKAN:

dataset/
├── train/
│   ├── angklung/
│   ├── gamelan/
│   ├── kolintang/
│   ├── sasando/
│   ├── gendang/
│   ├── rebab/
│   ├── kecapi/
│   ├── tifa/
│   ├── bonang/
│   └── suling/
└── validation/
    ├── angklung/
    ├── gamelan/
    ├── kolintang/
    ├── sasando/
    ├── gendang/
    ├── rebab/
    ├── kecapi/
    ├── tifa/
    ├── bonang/
    └── suling/

Setiap folder kelas diisi gambar alat musik tersebut (disarankan minimal
100-200 gambar per kelas untuk hasil yang lebih baik).

Cara menjalankan:
    python train_model.py
"""

import os
import json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==============================
# KONFIGURASI
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'validation')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'vgg16_alat_musik.h5')
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(num_classes):
    """Membangun model dengan pendekatan Feature Extraction dari VGG16.
    Menggunakan GlobalAveragePooling2D (bukan Flatten) supaya jumlah
    parameter jauh lebih kecil -> model lebih ringan & hemat RAM saat
    deploy (penting untuk platform dengan limit memori seperti Railway
    free/hobby plan)."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)  # (7,7,512) -> (512,)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("[ERROR] Folder dataset/train atau dataset/validation belum ditemukan.")
        print("Silakan siapkan dataset sesuai struktur yang dijelaskan di bagian atas file ini.")
        return

    # Augmentasi data untuk training, hanya rescale untuk validasi
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = train_generator.num_classes
    print(f"[INFO] Jumlah kelas terdeteksi: {num_classes}")
    print(f"[INFO] Kelas: {train_generator.class_indices}")

    # Simpan mapping label -> index supaya app.py bisa menampilkan nama kelas yang benar
    with open(CLASS_INDEX_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)

    model = build_model(num_classes)
    model.summary()

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )

    print(f"[INFO] Training selesai. Model terbaik disimpan di {MODEL_PATH}")


if __name__ == '__main__':
    main()
