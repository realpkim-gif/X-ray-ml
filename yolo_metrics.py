import os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

# Scores a trained YOLO model in two ways:
#   1. per IMAGE: "is there a fracture in this X-ray?"  -> accuracy, precision, recall, f1, AUROC + ROC curve
#   2. per BOX:   "did it find each fracture in the right place?"  -> YOLO's own precision, recall, f1, mAP
#
#   from ultralytics import YOLO
#   from yolo_metrics import yolo_metrics
#   model = YOLO("runs/detect/bone_fracture_yolo/weights/best.pt")
#   yolo_metrics(model, "BoneData/BoneFractureYolo8/ModelWeights.yaml", "BoneData/BoneFractureYolo8/test/images", "test")


def yolo_metrics(model, data_yaml, image_folder, split="test", threshold=0.25, imgsz=1024):
    # image_folder: the split's images. Labels are found the YOLO way: .../images/x.jpg -> .../labels/x.txt

    # ---------- 1. per image ----------
    true = []     # 1 = the label file has at least one fracture box, 0 = empty (normal X-ray)
    scores = []   # the model's most confident box in the image (0 if it found nothing)

    for file in sorted(os.listdir(image_folder)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_folder, file)
        label_path = os.path.splitext(image_path.replace("images", "labels"))[0] + ".txt"

        has_fracture = 0
        if os.path.exists(label_path):
            with open(label_path) as f:
                if f.read().strip() != "":
                    has_fracture = 1
        true.append(has_fracture)

        # conf=0.001 keeps even weak boxes, so the ROC curve can use every possible threshold
        result = model.predict(image_path, imgsz=imgsz, conf=0.001, verbose=False)[0]
        if len(result.boxes) > 0:
            scores.append(float(result.boxes.conf.max()))
        else:
            scores.append(0.0)

    # a fracture is predicted when the best box is at least as confident as the threshold
    predicted = []
    for score in scores:
        if score >= threshold:
            predicted.append(1)
        else:
            predicted.append(0)

    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()

    def find_distance(true, scores):
        # average squared gap between the truth (0 or 1) and the model's confidence (0-1).
        # 0 = perfect, 1 = confidently wrong every time. (also called the Brier score)
        total = 0
        for i in range(len(true)):
            error = (true[i] - scores[i]) ** 2     # ** = power in Python (^ means something else)
            total += error
        return total / len(true)

    image_scores = {
        "accuracy": accuracy_score(true, predicted),
        "precision": precision_score(true, predicted, zero_division=0),
        "recall": recall_score(true, predicted, zero_division=0),         # share of fracture X-rays caught
        "specificity": tn / max(tn + fp, 1),                                # share of normal X-rays called normal
        "f1": f1_score(true, predicted, zero_division=0),
        "auroc": roc_auc_score(true, scores),                               # threshold-free, 0.5 = guessing, 1 = perfect
        "true_pos": tp, "false_pos": fp, "true_neg": tn, "false_neg": fn,
        "Euclidean_error": find_distance(true, scores),
    }


    # ROC curve: for every threshold, how many fractures caught vs how many normals wrongly flagged
    fpr, tpr, _ = roc_curve(true, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"YOLO (AUROC = {image_scores['auroc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random guessing")
    plt.xlabel("False positive rate (normals flagged)")
    plt.ylabel("True positive rate (fractures caught)")
    plt.title(f"ROC curve - {split}")
    plt.legend()
    plt.savefig(f"yolo_roc_{split}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- 2. per box (YOLO's own validation) ----------
    val = model.val(data=data_yaml, split=split, imgsz=imgsz, verbose=False)
    precision, recall = val.box.mp, val.box.mr
    box_scores = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "mAP50": val.box.map50,          # box counts as right if it overlaps the real one by 50%+
        "mAP50-95": val.box.map,         # stricter: averaged over 50% ... 95% overlap
    }

    print(f"\n{split} - per image (threshold {threshold}):")
    for name in image_scores:
        print(f"  {name:12} {round(float(image_scores[name]), 3)}")
    print(f"{split} - per box:")
    for name in box_scores:
        print(f"  {name:12} {round(float(box_scores[name]), 3)}")
    print(f"ROC curve saved to yolo_roc_{split}.png")

    return image_scores, box_scores


# runs only when you Run/Debug this file directly (not when another file imports it)
if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("PATH/TO/runs/detect/YOUR_RUN/weights/best.pt")   # your trained detect model
    yolo_metrics(model, "human_data/combined/ModelWeights.yaml", "human_data/combined/images/test", "test")
