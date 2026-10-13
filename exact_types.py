import os
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Runs the fracture-type YOLO (ExactLocator.py) INSIDE the rough yellow boxes from vote.py:
#   1. take every X-ray that vote.py called "fracture" and its yellow box
#   2. crop the X-ray to that box + CROP_MARGIN (the same margin YOLO was trained with)
#   3. YOLO finds the precise fracture box + its type (diaphysis / epiphysis / metaphysis) inside the crop
#   4. the precise box is moved back to full-image pixels (1024 x 1024)
# Saves fracture_types_{split}.csv and pictures in results/types_{split}/
#   (yellow = vote.py's rough box, red = YOLO's precise box + type)
#
# Note: YOLO learned the types on HUMAN child wrists. Dogs/cats have no type labels, so these
# types can't be checked on animals yet. YOLO's own scores on human crops are printed at the end.

MODEL = "runs/detect/fracture_type_yolo/weights/best.pt"
CROP_MARGIN = 0.15          # same as combine_human.py
MIN_CONF = 0.25             # YOLO boxes below this confidence are ignored
IMG_SIZE = 640              # same as training

model = YOLO(MODEL)


def read_box(text):
    # vote.py saves the box as text "[x1, y1, x2, y2]"; empty for "normal"
    if not isinstance(text, str) or text.strip() == "":
        return None
    numbers = text.strip("[]").split(",")
    return [float(n) for n in numbers]


for split in ["val", "test"]:
    votes = pd.read_csv(f"vote_{split}_predictions.csv")
    os.makedirs(f"results/types_{split}", exist_ok=True)
    rows = []

    for i in range(len(votes)):
        image_path = votes["image_path"][i]
        yellow = read_box(votes["box"][i])
        fracture_type, type_conf, precise_box, found = "", "", "", 0

        if votes["predicted"][i] == "fracture" and yellow is not None:
            img = Image.open(image_path).convert("RGB")

            # crop = yellow box + margin, kept inside the image
            x1, y1, x2, y2 = yellow
            pad_x = (x2 - x1) * CROP_MARGIN
            pad_y = (y2 - y1) * CROP_MARGIN
            cx1 = max(0, round(x1 - pad_x))
            cy1 = max(0, round(y1 - pad_y))
            cx2 = min(img.width, round(x2 + pad_x))
            cy2 = min(img.height, round(y2 + pad_y))
            crop = img.crop((cx1, cy1, cx2, cy2))

            result = model.predict(crop, imgsz=IMG_SIZE, conf=MIN_CONF, verbose=False)[0]
            found = len(result.boxes)
            if found == 0:
                fracture_type = "not found"
            else:
                best = int(result.boxes.conf.argmax())                 # YOLO's most confident box
                fracture_type = result.names[int(result.boxes.cls[best])]
                type_conf = round(float(result.boxes.conf[best]), 3)
                bx1, by1, bx2, by2 = [float(v) for v in result.boxes.xyxy[best]]   # pixels inside the crop
                precise_box = [round(bx1 + cx1), round(by1 + cy1), round(bx2 + cx1), round(by2 + cy1)]   # full image

            # picture: yellow rough box, red precise box + type
            draw = ImageDraw.Draw(img)
            draw.rectangle(yellow, outline=(255, 255, 0), width=4)
            if precise_box != "":
                draw.rectangle(precise_box, outline=(255, 0, 0), width=4)
                draw.text((precise_box[0], max(0, precise_box[1] - 14)), f"{fracture_type} {type_conf}", fill=(255, 0, 0))
            img.save(f"results/types_{split}/{os.path.basename(image_path)}")

        rows.append([image_path, votes["true"][i], votes["predicted"][i], votes["confidence"][i],
                     votes["box"][i], fracture_type, type_conf, precise_box, found])

    results = pd.DataFrame(rows, columns=["image_path", "true", "vote_predicted", "vote_confidence", "yellow_box",
                                          "fracture_type", "type_confidence", "precise_box", "yolo_boxes_found"])
    results.to_csv(f"fracture_types_{split}.csv", index=False)
    print(split, "types:", results["fracture_type"][results["fracture_type"] != ""].value_counts().to_dict())

# YOLO's own scores on the HUMAN test crops (precise box + type), per type
scores = model.val(data="human_data/crops/data.yaml", split="test", imgsz=IMG_SIZE, verbose=False)
print("\nYOLO on human test crops: precision", round(float(scores.box.mp), 3), "recall", round(float(scores.box.mr), 3),
      "mAP50", round(float(scores.box.map50), 3), "mAP50-95", round(float(scores.box.map), 3))
for k, name in scores.names.items():
    if k < len(scores.box.ap50):
        print(f"  {name:12} mAP50 {scores.box.ap50[k]:.3f}")
