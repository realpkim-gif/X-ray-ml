import numpy as np
import pandas as pd
import os
import ultralytics
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO("yolo26x-obb.pt")
results = model.train(
    data="BoneData/BoneFractureYolo8/data.yaml",
    epochs=100,
    imgsz=1024,
    patience=10,  # early stopping
    batch=4,  # tweak based on GPU memory
    name="bone_fracture_yolo",
    workers=4,
    device=0,
)

def yolo_predict_score(path):
    return path

    #TODO: We need to do this later