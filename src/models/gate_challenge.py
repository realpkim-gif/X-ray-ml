import sys
import os
import random
import datetime
import time
import pandas as pd
from torchvision import datasets
from PIL import Image
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)
# Python only finds other files in the SAME folder. This adds the folders whose files we import:
sys.path.append(os.path.join(PROJECT_FOLDER, "src", "models"))

import gate_extra                       # makes the generated pictures (screenshots, X-ray variations, ...)
from Xray_Or_Not import load_head, xray_chance

# ======================================================================================
# One CHALLENGE round for the "X-ray or not" gate: 10 NEW pictures it has never seen.
#   5 X-ray pictures  (should get "X-ray")
#   5 other pictures  (should get "not X-ray")
# Each picture's type is picked at random from the lists below, with a new random seed every round.
# The 10 pictures are saved in gate_challenge/round_<number>/ so you can look at them,
# and every result is added as a row to gate_challenge_results.csv.
#
#   python src/models/gate_challenge.py
# ======================================================================================

RESULTS_FILE = "metrics/gate_challenge_results.csv"
CHALLENGE_FOLDER = "metrics/gate_challenge"
WEB_FOLDER = "data/challenge_web"            # real pictures from the internet: challenge_web/1/ = X-rays, challenge_web/0/ = not
PHOTOS_FOLDER = "data/gate_photos"
USED_FILE = "data/gate_used_photos.csv"      # photos the gate was trained/tested on (we skip those)

# X-ray types: "xray_full" = an untouched test X-ray, "web_xray" = a real X-ray from the internet,
# the rest are X-ray variations from gate_extra.py (always made from TEST-split X-rays)
XRAY_TYPES = ["xray_full", "web_xray", "xray_crop", "xray_rotated", "xray_inverted", "xray_small",
              "xray_low_contrast", "xray_annotated", "xray_photo"]
# not X-ray types: "unseen_photo" / "unseen_gray_photo" = Caltech-256 / Places365 photos the gate never saw,
# "web_other" = a real random picture from the internet, the rest are made by gate_extra.py
OTHER_TYPES = ["unseen_photo", "unseen_gray_photo", "web_other", "text_page", "code", "ui_layout",
               "chart", "table", "drawing", "noise"]


def web_pictures(label):
    # all internet pictures of one label (empty list if the folder doesn't exist)
    folder = f"{WEB_FOLDER}/{label}"
    if not os.path.exists(folder):
        return []
    paths = []
    for file in sorted(os.listdir(folder)):
        ending = file.lower().split(".")[-1]
        if ending in ["jpg", "jpeg", "png", "webp"]:        # only real picture files
            paths.append(f"{folder}/{file}")
    return paths


def unseen_photo(rng):
    # a Caltech-256 or Places365 photo that was NOT used to train or test the gate
    used = pd.read_csv(USED_FILE)
    while True:
        if rng.random() < 0.5:
            source = "caltech256"
            dataset = datasets.Caltech256(root=PHOTOS_FOLDER, download=False)
        else:
            source = "places365"
            dataset = datasets.Places365(root=f"{PHOTOS_FOLDER}/places365", split="val", small=True, download=False)
        number = rng.randint(0, len(dataset) - 1)
        already_used = ((used["source"] == source) & (used["number"] == number)).any()
        if not already_used:
            return dataset[number][0]


def make_picture(kind, rng):
    # one picture of this type
    if kind == "xray_full":
        return Image.open(rng.choice(gate_extra.xray_paths("test")))
    if kind == "web_xray":
        return Image.open(rng.choice(web_pictures(1)))
    if kind == "web_other":
        return Image.open(rng.choice(web_pictures(0)))
    if kind == "unseen_photo":
        return unseen_photo(rng)
    if kind == "unseen_gray_photo":
        return unseen_photo(rng).convert("L")
    make = gate_extra.TYPES[kind][0]
    return make(rng, "test")                        # generated pictures always use TEST-split X-rays


# which round is this?
if os.path.exists(RESULTS_FILE):
    old_results = pd.read_csv(RESULTS_FILE)
    round_number = old_results["round"].max() + 1
else:
    old_results = None
    round_number = 1

# a new seed every round -> new pictures. It comes from the clock (time in nanoseconds), NOT from random:
# importing Xray_Or_Not.py sets random.seed(42), which would give the same "random" pictures every round
seed = time.time_ns() % 10**9
rng = random.Random(seed)
folder = f"{CHALLENGE_FOLDER}/round_{round_number}"
os.makedirs(folder, exist_ok=True)

# leave out internet types if no internet pictures were downloaded
xray_types = list(XRAY_TYPES)
if len(web_pictures(1)) == 0:
    xray_types.remove("web_xray")
other_types = list(OTHER_TYPES)
if len(web_pictures(0)) == 0:
    other_types.remove("web_other")

head = load_head()
time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
rows = []
for i in range(10):
    if i < 5:
        kind = rng.choice(xray_types)
        truth = 1
    else:
        kind = rng.choice(other_types)
        truth = 0
    picture = make_picture(kind, rng).convert("RGB")
    path = f"{folder}/{i + 1}_{kind}.png"
    picture.save(path)

    chance = xray_chance(head, picture)
    if chance >= 0.5:
        predicted = 1
    else:
        predicted = 0
    rows.append({"round": round_number, "time": time_now, "seed": seed, "picture": path, "type": kind,
                 "truth": truth, "xray_chance": round(chance, 4), "predicted": predicted,
                 "correct": predicted == truth, "counts": True})

results = pd.DataFrame(rows)
print(results[["type", "truth", "xray_chance", "predicted", "correct"]].to_string(index=False))
print(f"round {round_number}: {int(results['correct'].sum())} / 10 correct")

if old_results is not None:
    results = pd.concat([old_results, results])
results.to_csv(RESULTS_FILE, index=False)
