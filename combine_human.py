"""Combine FracAtlas + GRAZPEDWRI-DX into one YOLO dataset with a single class: fracture.

Output (human_data/combined):
    images/train, images/val, images/test     8-bit PNG / JPG X-rays
    labels/train, labels/val, labels/test     one .txt per image: "0 x_center y_center width height" (0-1)
                                              empty .txt = no fracture (a normal image, still useful)
    data.yaml                                 for YOLO training
    images.csv                                one row per image: source, patient, split, number of fractures

Split 70/15/15 by PATIENT, so all X-rays of one person land in the same split.
Both datasets are CC BY 4.0: credit them if you use this (see CREDITS at the bottom).

    python combine_human.py
"""
import os
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image

RAW = "human_data/raw"
OUT = "human_data/combined"
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SEED = 42

FRACATLAS = f"{RAW}/FracAtlas/FracAtlas"
GRAZ_LABELS = f"{RAW}/GRAZ/yolov5/labels"
GRAZ_IMAGES = f"{RAW}/GRAZ_images"          # where images_part1-4.zip get unzipped
GRAZ_CSV = "human_data/zips/dataset.csv"
GRAZ_FRACTURE = 3                           # "fracture" in GRAZ's meta.yaml (0 boneanomaly ... 8 text)

for split in ["train", "val", "test"]:
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)


def read_boxes(path, keep_class):
    # YOLO label file -> list of "x y w h" strings for one class only (all other classes dropped)
    if not os.path.exists(path):
        return []
    boxes = []
    with open(path) as f:
        for line in f.read().split("\n"):
            parts = line.split()
            # FracAtlas files sometimes have 2 boxes on one line (no newline), so read 5 numbers at a time
            for k in range(0, len(parts) - 4, 5):
                if int(float(parts[k])) == keep_class:
                    boxes.append(" ".join(parts[k + 1:k + 5]))
    return boxes


# 1. FracAtlas: no patient ids, so every image counts as its own patient
rows = []
for folder in ["Fractured", "Non_fractured"]:
    for file in sorted(os.listdir(f"{FRACATLAS}/images/{folder}")):
        stem = os.path.splitext(file)[0]
        rows.append({"source": "fracatlas", "image": f"{FRACATLAS}/images/{folder}/{file}",
                     "patient": f"fracatlas_{stem}", "name": f"fracatlas_{file}",
                     "boxes": read_boxes(f"{FRACATLAS}/Annotations/YOLO/{stem}.txt", keep_class=0)})
print("FracAtlas images:", len(rows))


# 2. GRAZPEDWRI-DX: patient id is the first part of the name, "0001_1297860395_01_WRI-L1_M014" -> "0001"
graz_files = {}                             # file stem -> full path (the zips may have sub-folders)
for folder, _, files in os.walk(GRAZ_IMAGES):
    for file in files:
        if file.endswith(".png"):
            graz_files[os.path.splitext(file)[0]] = os.path.join(folder, file)

graz = pd.read_csv(GRAZ_CSV, encoding="utf-8-sig", dtype=str)
missing = 0
for stem in graz["filestem"]:
    if stem not in graz_files:
        missing += 1
        continue
    rows.append({"source": "grazpedwri", "image": graz_files[stem],
                 "patient": "graz_" + stem.split("_")[0], "name": f"graz_{stem}.png",
                 "boxes": read_boxes(f"{GRAZ_LABELS}/{stem}.txt", keep_class=GRAZ_FRACTURE)})
print("GRAZPEDWRI-DX images:", len(graz) - missing, "| missing image files:", missing)
data = pd.DataFrame(rows)


# 3. Split by patient (70 / 15 / 15). Patients with and without fractures are split
#    separately, so every split gets the same mix.
data["fractured"] = data["boxes"].apply(len) > 0
patients = data.groupby("patient")["fractured"].any().reset_index()
random.seed(SEED)
split_of = {}
for _, group in patients.groupby("fractured"):
    names = list(group["patient"])
    random.shuffle(names)
    n_test = round(len(names) * TEST_SIZE)
    n_val = round(len(names) * VAL_SIZE)
    for i, name in enumerate(names):
        split_of[name] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
data["split"] = data["patient"].map(split_of)


# 4. Copy images + write labels. GRAZ PNGs are 16-bit: most tools (and PIL .convert("RGB"))
#    would show them almost all white, so they are stretched to normal 8-bit (0-255) first.
def to_8bit(path, new_path):
    with Image.open(path) as img:
        pixels = np.array(img).astype(np.float32)
    if pixels.max() > 255:                  # 16-bit image
        low, high = np.percentile(pixels, [0.5, 99.5])   # ignore a few extreme pixels
        pixels = np.clip((pixels - low) / max(high - low, 1) * 255, 0, 255)
    Image.fromarray(pixels.astype(np.uint8)).save(new_path)

for i, row in enumerate(data.itertuples()):
    image_path = f"{OUT}/images/{row.split}/{row.name}"
    label_path = f"{OUT}/labels/{row.split}/{os.path.splitext(row.name)[0]}.txt"
    if not os.path.exists(image_path):      # skip ones already done, so a re-run continues
        if row.source == "grazpedwri":
            to_8bit(row.image, image_path)
        else:
            shutil.copy(row.image, image_path)
    with open(label_path, "w") as f:
        f.write("\n".join(f"0 {box}" for box in row.boxes))   # class 0 = fracture
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} / {len(data)} done")


# 5. data.yaml for YOLO + a summary table
with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "nc: 1\nnames: ['fracture']\n")

data["fractures"] = data["boxes"].apply(len)
data[["name", "source", "patient", "split", "fractures"]].to_csv(f"{OUT}/images.csv", index=False)
print(data.groupby(["split", "source"]).agg(images=("name", "count"), with_fracture=("fractured", "sum"),
                                            boxes=("fractures", "sum")))

CREDITS = """
FracAtlas (CC BY 4.0): Abedeen et al., "FracAtlas: A Dataset for Fracture Classification, Localization
  and Segmentation of Musculoskeletal Radiographs", Scientific Data 2023. figshare 10.6084/m9.figshare.22363012
GRAZPEDWRI-DX (CC BY 4.0): Nagy et al., "A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for
  machine learning", Scientific Data 2022. figshare 10.6084/m9.figshare.14825193
"""
with open(f"{OUT}/CREDITS.txt", "w") as f:
    f.write(CREDITS)
