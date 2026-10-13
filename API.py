# ======================================================================================
# API.py - the Radiant fracture API
#
# What is an API?  A program that waits for requests from other programs (here: your
# website) and sends back answers. The website sends an X-ray image, this file runs the
# whole model pipeline on it, and sends back JSON (text the website can read), e.g.
#   {"answer": "fracture", "confidence": 0.9, "box": {...}, "fracture_type": "metaphysis"}
#
# How to start it (from the project folder, in a terminal):
#   python -m uvicorn API:app --host 127.0.0.1 --port 8000
# Then open http://127.0.0.1:8000/docs in a browser: FastAPI makes a test page there
# where you can upload an X-ray and see the answer, without any website.
#
# The pipeline for ONE uploaded X-ray (the same steps as training, so the models see
# the same kind of input they learned on):
#   1. resize to 1024 x 1024                       (the training X-rays were 1024 x 1024)
#   2. ConvNeXt  -> cnn vector [1, 1024]           (like Pretrained_CNN.py)
#   3. DINO      -> dino vector [1, 1536]          (like DINO.py: summary + average)
#                -> patches [1, 32, 32, 768]       (like DINO_patches.py: every square)
#   4. CNN+DINO head (head.pt) 5x with dropout ON  -> 5 votes   (Monte Carlo dropout, like vote.py)
#      patch head (fracture_head.pt) 5x            -> 5 votes + 5 heatmaps
#   5. majority of the 10 votes = answer, share agreeing = confidence
#   6. average heatmap -> rough yellow box         (like FractureLocator.py)
#   7. YOLO (best.pt) inside the yellow box        -> precise box + fracture type (like exact_types.py)
# ======================================================================================

# ---------- imports: code from other libraries / files that we use ----------
import io                                    # io.BytesIO turns the uploaded bytes into a "file" PIL can open
import json                                  # to read thresholds.json
import pandas as pd                          # to read fracture_metrics.csv
import torch                                 # PyTorch: runs the models
import timm                                  # library that has the ConvNeXt model
from PIL import Image                        # opens and resizes images
from transformers import AutoImageProcessor, AutoModel   # library that has the DINOv2 model
from ultralytics import YOLO                 # library that has YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException   # the API library (explained below)
from fastapi.middleware.cors import CORSMiddleware            # lets the website (another address) call us
import main                                  # our main.py: Head (CNN+DINO head), label_map, device
import FractureLocator as fl                 # our FractureLocator.py: PatchHead, find_box, image_scores


# ---------- settings (CAPITALS = values you might want to change) ----------
PASSES = 5                                   # Monte Carlo dropout passes per head (5 + 5 = 10 votes)
DINO_SIZE = 448                              # same as DINO.py / DINO_patches.py
YOLO_WEIGHTS = "runs/detect/fracture_type_yolo/weights/best.pt"   # best YOLO epoch (best.pt, not last.pt)
YOLO_SIZE = 640                              # same as ExactLocator.py training
YOLO_MIN_CONF = 0.25                         # ignore YOLO boxes below this confidence
CROP_MARGIN = 0.15                           # same margin as combine_human.py / exact_types.py
MAX_UPLOAD_MB = 15                           # refuse files bigger than this (protects the server)
# Which websites may call this API. A browser blocks calls from other addresses unless we list them here.
ALLOWED_WEBSITES = [
    "https://realpkim-gif.github.io",        # your GitHub Pages site
    "http://localhost:5500",                 # VS Code "Live Server" while you test the website locally
    "http://127.0.0.1:5500",
]
device = main.device                         # "cuda" (the GPU) if there is one, else "cpu"


# ======================================================================================
# 1. LOAD EVERYTHING ONCE, when the API starts.
#    Loading models takes seconds; doing it for every request would be very slow.
#    These variables stay in memory and every request reuses them.
# ======================================================================================

# ConvNeXt (frozen backbone #1) - exactly like Pretrained_CNN.py
cnn_model = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True, num_classes=0)
cnn_model = cnn_model.eval().to(device)      # eval = inference mode (this backbone has no dropout we want)
cnn_config = timm.data.resolve_model_data_config(cnn_model)
cnn_transform = timm.data.create_transform(**cnn_config, is_training=False)   # its own resize + normalisation

# DINOv2 (frozen backbone #2) - exactly like DINO.py: 448 x 448, no centre crop
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base",
                                                    size={"height": DINO_SIZE, "width": DINO_SIZE},
                                                    do_center_crop=False)
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)

# Our two trained heads. load_state_dict() puts the saved, trained numbers into an empty model.
fracture_column = main.label_map["fracture"]                     # which output of Head means "fracture" (0)
cnn_dino_head = main.Head(1024, 1536, main.num_classes).to(device)
cnn_dino_head.load_state_dict(torch.load("head.pt", map_location=device))
patch_head = fl.PatchHead(768).to(device)
patch_head.load_state_dict(torch.load("fracture_head.pt", map_location=device))

# The two thresholds the heads vote with. Both were chosen on VAL to catch 90% of fractures:
#   CNN+DINO head -> thresholds.json (written by main.py)
#   patch head    -> the "threshold" column of fracture_metrics.csv (written by FractureLocator.py)
with open("thresholds.json") as f:
    cnn_dino_threshold = json.load(f)["fracture"]
patch_threshold = float(pd.read_csv("fracture_metrics.csv")["threshold"][0])

# YOLO: precise box + fracture type inside the yellow box
yolo_model = YOLO(YOLO_WEIGHTS)

print("models loaded on", device)


# ======================================================================================
# 2. THE PIPELINE FOR ONE IMAGE (a normal Python function, the API calls it below)
# ======================================================================================

def to_fraction(box, width, height):
    # [x1, y1, x2, y2] in pixels -> 0-1 fractions of the image size.
    # Fractions are easiest for the website: multiply by the shown image size to draw the box,
    # whatever size the image is displayed at.
    return {"x1": round(box[0] / width, 4), "y1": round(box[1] / height, 4),
            "x2": round(box[2] / width, 4), "y2": round(box[3] / height, 4)}


def analyse(image):
    # image = PIL image as uploaded. Returns a dictionary (becomes the JSON answer).

    # --- step 1: same size as training ---
    image = image.convert("RGB").resize((1024, 1024))   # gray X-ray -> 3 channels, 1024 x 1024

    with torch.no_grad():                    # no training here, so no gradients (faster, less memory)
        # --- step 2: ConvNeXt vector ---
        cnn = cnn_model(cnn_transform(image).unsqueeze(0).to(device))   # [1, 1024]

        # --- step 3: DINO. ONE pass gives both the image vector and the square vectors ---
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        tokens = dino_model(**inputs).last_hidden_state        # [1, 1 + 1024, 768]
        summary = tokens[:, 0]                                  # CLS token          [1, 768]
        average = tokens[:, 1:].mean(dim=1)                     # mean of squares    [1, 768]
        dino = torch.cat([summary, average], dim=1)             # like DINO.py       [1, 1536]
        patches = tokens[:, 1:].reshape(1, 32, 32, 768).half()  # like DINO_patches  [1, 32, 32, 768]

        # --- step 4: Monte Carlo dropout. train() = dropout ON, so every pass differs a little ---
        cnn_dino_head.train()
        patch_head.train()
        votes = []                           # 10 yes(1)/no(0) votes
        heatmaps = []                        # 5 heatmaps [32, 32] from the patch head
        for _ in range(PASSES):
            prob = torch.sigmoid(cnn_dino_head(cnn, dino))[0, fracture_column]
            votes.append(int(prob >= cnn_dino_threshold))
        for _ in range(PASSES):
            square_scores = patch_head(patches)                 # [1, 32, 32]
            prob = torch.sigmoid(fl.image_scores(square_scores))[0]
            votes.append(int(prob >= patch_threshold))
            heatmaps.append(torch.sigmoid(square_scores)[0].cpu())

    # --- step 5: majority vote ---
    fracture_share = sum(votes) / len(votes)                    # e.g. 8 of 10 -> 0.8
    is_fracture = fracture_share >= 0.5                         # 5-5 tie counts as fracture
    confidence = max(fracture_share, 1 - fracture_share)        # share agreeing with the answer

    result = {
        "answer": "fracture" if is_fracture else "normal",
        "confidence": round(confidence, 2),
        "fracture_votes": f"{sum(votes)} of {len(votes)}",
        "rough_box": None,                   # None becomes null in JSON = "no box"
        "precise_box": None,
        "fracture_type": None,
        "type_confidence": None,
        "disclaimer": "Not a diagnosis. Always have a veterinarian review the X-ray.",
    }
    if not is_fracture:
        return result                        # normal: no boxes, done

    # --- step 6: rough yellow box from the average heatmap ---
    mean_heatmap = torch.stack(heatmaps).mean(0).numpy()       # [32, 32]
    x1, y1, x2, y2 = fl.find_box(mean_heatmap)                  # pixels of the 1024 x 1024 image
    result["rough_box"] = to_fraction([x1, y1, x2, y2], 1024, 1024)

    # --- step 7: YOLO inside the yellow box (+ margin), like exact_types.py ---
    pad_x = (x2 - x1) * CROP_MARGIN
    pad_y = (y2 - y1) * CROP_MARGIN
    cx1, cy1 = max(0, round(x1 - pad_x)), max(0, round(y1 - pad_y))
    cx2, cy2 = min(1024, round(x2 + pad_x)), min(1024, round(y2 + pad_y))
    crop = image.crop((cx1, cy1, cx2, cy2))
    found = yolo_model.predict(crop, imgsz=YOLO_SIZE, conf=YOLO_MIN_CONF, verbose=False)[0]
    if len(found.boxes) > 0:
        best = int(found.boxes.conf.argmax())                   # YOLO's most confident box
        bx1, by1, bx2, by2 = [float(v) for v in found.boxes.xyxy[best]]   # pixels inside the crop
        precise = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]  # move back to the 1024 image
        result["precise_box"] = to_fraction(precise, 1024, 1024)
        result["fracture_type"] = found.names[int(found.boxes.cls[best])]
        result["type_confidence"] = round(float(found.boxes.conf[best]), 2)
    return result


# ======================================================================================
# 3. THE API ITSELF (FastAPI)
#    app = the API object. Uvicorn (the server program) runs it: "uvicorn API:app".
#    @app.get / @app.post = "when a request comes to this address, run the function below".
#      GET  = ask for something (like opening a web page)
#      POST = send something (like uploading a file)
# ======================================================================================

app = FastAPI(title="Radiant fracture API",
              description="Upload a dog or cat limb X-ray: fracture yes/no, confidence, boxes, type.")

# CORS: browsers only let a web page call a DIFFERENT address (our API) if that address says it's allowed.
# Your website (github.io) and this API (another address) are different, so we list the website here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_WEBSITES,          # who may call us
    allow_methods=["GET", "POST"],           # which kinds of requests
    allow_headers=["*"],
)


@app.get("/")
def health():
    # Open http://127.0.0.1:8000/ in a browser -> you see this. Handy to check the API is running.
    return {"status": "ok", "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # The website sends the X-ray here as a "multipart" upload (a normal file upload).
    # UploadFile = the uploaded file. File(...) = it is required (... means "must be given").
    # async = FastAPI can keep accepting other requests while this one waits for the upload to arrive.

    data = await file.read()                 # the raw bytes of the file
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        # HTTPException = send an error back instead of an answer. 413 = "too large".
        raise HTTPException(status_code=413, detail=f"File is bigger than {MAX_UPLOAD_MB} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()                         # actually decode it now, so broken files fail HERE
    except Exception:
        # 400 = "bad request": the client sent something we can't use
        raise HTTPException(status_code=400, detail="That file is not an image we can open")

    return analyse(image)                    # FastAPI turns the dictionary into JSON automatically
