import random
import torch
import pandas as pd
import numpy as np
import FractureLocator as fl   # same settings, model and scores as FractureLocator.py

# Honest score for FractureLocator. val/test only have ~10 animals each, so one split is very noisy.
# 5-fold cross-validation BY ANIMAL on all ModelWeights (train + val + test together):
#   split the animals into 5 groups -> train on 4 groups, score the 5th -> repeat 5 times.
#   every animal is scored once, always by a head that never saw that animal.
# Mirrored copies are only used for training; scores are measured on the real images.
# Animals from other sources (all fractures) are always in the training part, never scored.

FOLDS = 5
SEED = 42

# everything in one table (train + val + test)
data_parts, patch_parts, mask_parts, label_parts = [], [], [], []
for split in ["train", "val", "test"]:
    data, patches, masks, labels = fl.load(split, keep_flipped=True)
    data_parts.append(data)
    patch_parts.append(patches)
    mask_parts.append(masks)
    label_parts.append(labels)
data = pd.concat(data_parts, ignore_index=True)
patches = torch.cat(patch_parts)    # [N, 32, 32, 768]
masks = torch.cat(mask_parts)       # [N, 32, 32]
labels = torch.cat(label_parts)     # [N]

# give every animal a fold number 0-4. Fracture animals and normal animals are dealt out
# separately (like cards), so every fold gets about the same share of fractures.
fracture_animals, normal_animals = [], []
for animal, group in data[data["view"] != "other_source"].groupby("animal"):
    if (group["label"] == "fracture").any():
        fracture_animals.append(animal)
    else:
        normal_animals.append(animal)
random.seed(SEED)
random.shuffle(fracture_animals)
random.shuffle(normal_animals)
fold_of = {}                         # animal -> fold number
for i, animal in enumerate(fracture_animals):
    fold_of[animal] = i % FOLDS
for i, animal in enumerate(normal_animals):
    fold_of[animal] = i % FOLDS
folds = []
for animal in data["animal"]:
    if animal in fold_of:
        folds.append(fold_of[animal])
    else:
        folds.append(float("nan"))           # other-source animals get no fold -> always training
data["fold"] = folds

all_scores = torch.zeros(masks.shape)        # square scores of every image, from the fold where it was held out
scored = np.zeros(len(data), dtype=bool)

for fold in range(FOLDS):
    is_test = (data["fold"] == fold).values
    train_rows = np.where(~is_test)[0]                                  # includes mirrored copies
    test_rows = np.where(is_test & (data["flipped"] == False).values)[0]   # real images only

    torch.manual_seed(SEED + fold)
    train_data = data.iloc[train_rows].reset_index(drop=True)
    model, _ = fl.train_head(train_data, patches[train_rows], masks[train_rows], labels[train_rows])
    all_scores[test_rows] = fl.predict_all(model, patches[test_rows])
    scored[test_rows] = True

    test_data = data.iloc[test_rows].reset_index(drop=True)
    probs = torch.sigmoid(fl.image_scores(all_scores[test_rows]))
    aurocs = fl.view_aurocs(test_data, labels[test_rows], probs)
    print(f"fold {fold + 1}: {len(test_data['animal'].unique())} animals, {len(test_rows)} images, "
          f"AUROC all {aurocs['all']:.3f} limb {aurocs.get('limb', float('nan')):.3f} body {aurocs.get('body', float('nan')):.3f}")

# all held-out images together = one score over every animal
rows = np.where(scored)[0]
cv_data = data.iloc[rows].reset_index(drop=True)
probs = torch.sigmoid(fl.image_scores(all_scores[rows]))
threshold = fl.pick_threshold(probs, labels[rows])
results, scores = fl.evaluate(cv_data, masks[rows], labels[rows], all_scores[rows], threshold)
results.to_csv("fracture_cv_predictions.csv", index=False)
print("\ncross-validated (all animals):", scores)
