"""Combine FracAtlas + GRAZPEDWRI-DX into one YOLO dataset with a single class: fracture.

Output (human_data/combined):
    images/train, images/val, images/test     8-bit PNG / JPG X-rays
    labels/train, labels/val, labels/test     one .txt per image: "0 x_center y_center width height" (0-1)
                                              empty .txt = no fracture (a normal image, still useful)
    ModelWeights.yaml                                 for YOLO training
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
                     "boxes": read_boxes(f"{FRACATLAS}/Annotations/YOLO/{stem}.txt", keep_class=0),
                     "ao": ""})                 # FracAtlas has no fracture types
print("FracAtlas images:", len(rows))


# 2. GRAZPEDWRI-DX: patient id is the first part of the name, "0001_1297860395_01_WRI-L1_M014" -> "0001"
graz_files = {}                             # file stem -> full path (the zips may have sub-folders)
for folder, _, files in os.walk(GRAZ_IMAGES):
    for file in files:
        if file.endswith(".png"):
            graz_files[os.path.splitext(file)[0]] = os.path.join(folder, file)

graz = pd.read_csv(GRAZ_CSV, encoding="utf-8-sig", dtype=str)
missing = 0
ao_codes = graz["ao_classification"].fillna("")   # fracture type, e.g. "23r-M/3.1; 23u-E/7"
for stem, ao in zip(graz["filestem"], ao_codes):
    if stem not in graz_files:
        missing += 1
        continue
    rows.append({"source": "grazpedwri", "image": graz_files[stem],
                 "patient": "graz_" + stem.split("_")[0], "name": f"graz_{stem}.png",
                 "boxes": read_boxes(f"{GRAZ_LABELS}/{stem}.txt", keep_class=GRAZ_FRACTURE),
                 "ao": ao})
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


# 5. ModelWeights.yaml for YOLO + a summary table
with open(f"{OUT}/ModelWeights.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "nc: 1\nnames: ['fracture']\n")

data["fractures"] = data["boxes"].apply(len)
data[["name", "source", "patient", "split", "fractures"]].to_csv(f"{OUT}/images.csv", index=False)
print(data.groupby(["split", "source"]).agg(images=("name", "count"), with_fracture=("fractured", "sum"),
                                            boxes=("fractures", "sum")))

# 6. Cropped dataset for fracture TYPE (human_data/crops).
#    Only GRAZ has types (AO codes). We keep only the letter after the dash = WHERE on the bone:
#      "23r-M/3.1" -> M = metaphysis (near the end), E = epiphysis (growth-plate end), D = diaphysis (shaft)
#    That works for any long bone, also in dogs. Images whose fractures are in different regions
#    (e.g. "23r-M/3.1; 23u-E/7") are skipped: the code doesn't say which box is which.
#    Each image is cropped to ONE box around all its fracture boxes (smallest x1/y1, largest x2/y2),
#    plus CROP_MARGIN on every side, like the rough yellow box from vote.py.
CROPS = "human_data/crops"
CROP_MARGIN = 0.15                          # 15% extra on each side (0 = exactly the outer box)
REGIONS = ["diaphysis", "epiphysis", "metaphysis"]   # YOLO class 0, 1, 2
LETTER_TO_CLASS = {"D": 0, "E": 1, "M": 2}

for split in ["train", "val", "test"]:
    os.makedirs(f"{CROPS}/images/{split}", exist_ok=True)
    os.makedirs(f"{CROPS}/labels/{split}", exist_ok=True)


def region_of(ao):
    # "23r-M/3.1; 23u-M/2.1" -> 2 (metaphysis). Different regions or no code -> None
    letters = set()
    for code in ao.split(";"):
        code = code.strip()
        if "-" in code:
            letters.add(code.split("-")[1][0])
    if len(letters) != 1:
        return None
    letter = letters.pop()
    if letter not in LETTER_TO_CLASS:
        return None
    return LETTER_TO_CLASS[letter]


crop_rows = []
for row in data.itertuples():
    if row.source != "grazpedwri" or len(row.boxes) == 0:
        continue
    region = region_of(row.ao)
    if region is None:
        continue

    with Image.open(f"{OUT}/images/{row.split}/{row.name}") as img:
        img = img.convert("L")
        width, height = img.size

        # YOLO boxes "x_center y_center w h" (0-1) -> pixel corners
        corners = []
        for box in row.boxes:
            x, y, w, h = [float(v) for v in box.split()]
            corners.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])

        # one outer box around all fracture boxes, plus the margin, kept inside the image
        x1 = min(c[0] for c in corners)
        y1 = min(c[1] for c in corners)
        x2 = max(c[2] for c in corners)
        y2 = max(c[3] for c in corners)
        pad_x = (x2 - x1) * CROP_MARGIN
        pad_y = (y2 - y1) * CROP_MARGIN
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        crop = img.crop((round(x1), round(y1), round(x2), round(y2)))

    # the fracture boxes again, but now relative to the crop (0-1)
    crop_w, crop_h = crop.size
    lines = []
    for c in corners:
        cx = ((c[0] + c[2]) / 2 - round(x1)) / crop_w
        cy = ((c[1] + c[3]) / 2 - round(y1)) / crop_h
        lines.append(f"{region} {cx:.6f} {cy:.6f} {(c[2] - c[0]) / crop_w:.6f} {(c[3] - c[1]) / crop_h:.6f}")

    stem = os.path.splitext(row.name)[0]
    crop.save(f"{CROPS}/images/{row.split}/{stem}.png")
    with open(f"{CROPS}/labels/{row.split}/{stem}.txt", "w") as f:
        f.write("\n".join(lines))
    crop_rows.append([stem, row.split, REGIONS[region], len(lines)])

with open(f"{CROPS}/data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(CROPS)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"nc: {len(REGIONS)}\nnames: {REGIONS}\n")
crops = pd.DataFrame(crop_rows, columns=["name", "split", "region", "boxes"])
crops.to_csv(f"{CROPS}/crops.csv", index=False)
print()
print("cropped dataset:")
print(crops.groupby(["split", "region"]).size())


CREDITS = """
FracAtlas (CC BY 4.0): Abedeen et al., "FracAtlas: A Dataset for Fracture Classification, Localization
  and Segmentation of Musculoskeletal Radiographs", Scientific Data 2023. figshare 10.6084/m9.figshare.22363012
GRAZPEDWRI-DX (CC BY 4.0): Nagy et al., "A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for
  machine learning", Scientific Data 2022. figshare 10.6084/m9.figshare.14825193
"""
with open(f"{OUT}/CREDITS.txt", "w") as f:
    f.write(CREDITS)
