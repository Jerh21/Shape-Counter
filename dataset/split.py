import os
import shutil
import random

def split_dataset():
    SOURCE_DIR = "data/all"
    DEST_ROOT = "data"
    TRAIN_DIR = os.path.join(DEST_ROOT, "train")
    VAL_DIR   = os.path.join(DEST_ROOT, "val")
    TEST_DIR  = os.path.join(DEST_ROOT, "test")

    TRAIN_RATIO = 0.7
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    all_images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpg")]
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    n_test  = n_total - n_train - n_val

    splits = [
        (TRAIN_DIR, all_images[:n_train]),
        (VAL_DIR,   all_images[n_train:n_train+n_val]),
        (TEST_DIR,  all_images[n_train+n_val:])
    ]

    for dest_dir, img_list in splits:
        for img_name in img_list:
            img_src = os.path.join(SOURCE_DIR, img_name)
            json_src = img_src.replace(".jpg", ".json")
            img_dst = os.path.join(dest_dir, img_name)
            json_dst = os.path.join(dest_dir, os.path.basename(json_src))
            shutil.copy2(img_src, img_dst)
            shutil.copy2(json_src, json_dst)

    print("¡Partición completada!")
    print(f"Entrenamiento: {len(os.listdir(TRAIN_DIR))//2} imágenes")
    print(f"Validación:    {len(os.listdir(VAL_DIR))//2} imágenes")
    print(f"Test:          {len(os.listdir(TEST_DIR))//2} imágenes")