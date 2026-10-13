from ultralytics import YOLO

# Trains YOLO on the CROPPED human wrist fractures made by combine_human.py (human_data/crops).
# Every crop = one rough box around the fracture (like vote.py's yellow box) plus a 15% margin.
# YOLO learns, inside such a crop:
#   - the PRECISE box around each fracture
#   - the fracture TYPE = where on the bone it is: diaphysis (shaft), epiphysis (growth-plate end),
#     metaphysis (near the end). From the GRAZPEDWRI-DX AO codes.
#
# Normal boxes, so the normal detect model ("yolo26x.pt"), not the rotated-box one ("-obb").
# Result: runs/detect/fracture_type_yolo/weights/best.pt  (used by exact_types.py)

if __name__ == "__main__":
    model = YOLO("yolo26x.pt")
    results = model.train(
        data="human_data/crops/data.yaml",
        epochs=100,
        imgsz=640,   # crops are at most 639 px, so 640 keeps every pixel (1024 would not fit in 8 GB)
        patience=10,  # early stopping: stop when val hasn't improved for 10 epochs
        batch=4,  # tweak based on GPU memory
        name="fracture_type_yolo",
        workers=4,
        device=0,
    )
