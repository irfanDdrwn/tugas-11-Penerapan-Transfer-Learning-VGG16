import os
import shutil
import random
from icrawler.builtin import BingImageCrawler


CLASSES = {
    "angklung": ["angklung alat musik", "angklung bambu indonesia"],
    "gamelan": ["gamelan jawa", "gamelan bali instrument"],
    "kolintang": ["kolintang alat musik", "kolintang sulawesi utara"],
    "sasando": ["sasando alat musik", "sasando rote ntt"],
    "gendang": ["gendang melayu", "kendang jawa alat musik"],
    "rebab": ["rebab alat musik", "rebab jawa"],
    "kecapi": ["kecapi sunda alat musik", "kecapi indonesia"],
    "tifa": ["tifa alat musik papua", "tifa maluku"],
    "bonang": ["bonang gamelan", "bonang alat musik jawa"],
    "suling": ["suling bambu indonesia", "suling sunda alat musik"],
}

IMAGES_PER_CLASS = 150       # target jumlah gambar per kelas (sebelum cleaning)
VAL_SPLIT = 0.2              # 20% untuk validasi, 80% untuk training

RAW_DIR = "dataset_raw"
TRAIN_DIR = os.path.join("dataset", "train")
VAL_DIR = os.path.join("dataset", "validation")


def download_images():
    os.makedirs(RAW_DIR, exist_ok=True)
    for class_name, keywords in CLASSES.items():
        class_dir = os.path.join(RAW_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        existing = len(os.listdir(class_dir))
        if existing >= IMAGES_PER_CLASS:
            print(f"[SKIP] '{class_name}' sudah punya {existing} gambar.")
            continue

        target_per_keyword = max(1, (IMAGES_PER_CLASS - existing) // len(keywords))

        print(f"\n[DOWNLOAD] Kelas: {class_name}")
        for kw in keywords:
            print(f"    -> mencari: '{kw}' ({target_per_keyword} gambar)")
            crawler = BingImageCrawler(
                storage={"root_dir": class_dir},
                log_level=50  # kurangi log yang terlalu ramai
            )
            crawler.crawl(keyword=kw, max_num=target_per_keyword)


def split_train_validation():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    for class_name in CLASSES.keys():
        raw_class_dir = os.path.join(RAW_DIR, class_name)
        if not os.path.exists(raw_class_dir):
            continue

        images = [f for f in os.listdir(raw_class_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        val_count = int(len(images) * VAL_SPLIT)
        val_images = images[:val_count]
        train_images = images[val_count:]

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(raw_class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(raw_class_dir, img), os.path.join(val_class_dir, img))

        print(f"[SPLIT] {class_name}: {len(train_images)} train, {len(val_images)} validation")


if __name__ == "__main__":
    print("=== TAHAP 1: Download gambar dari internet ===")
    download_images()

    print("\n=== TAHAP 2: Membagi ke folder train/validation ===")
    split_train_validation()

    print("\n✅ Selesai! Sekarang:")
    print("1. Buka folder dataset/train dan dataset/validation")
    print("2. Hapus manual gambar yang tidak relevan/buram/salah")
    print("3. Jalankan: python train_model.py")
