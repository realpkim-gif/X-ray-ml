# ======================================================================================
# API.py - the Radiant fracture API
#
# What is an API?  A program that waits for requests from other programs (here: your
# website) and sends back answers. The website sends an X-ray image, this file runs the
# whole model pipeline on it, and sends back JSON (text the website can read), e.g.
#   {"answer": "fracture", "confidence": 0.9, "box": {...}}
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
#   6. average heatmap -> yellow box               (like FractureLocator.py)
#   7. draw the heatmap and/or the box on the ORIGINAL upload, as the caller asked:
#        POST /predict?show=box       only the yellow box
#        POST /predict?show=heatmap   only the red heatmap
#        POST /predict?show=both      both (the default)
# ======================================================================================

# ---------- imports: code from other libraries / files that we use ----------
import io                                    # io.BytesIO turns the uploaded bytes into a "file" PIL can open
import base64                                # turns the finished picture (bytes) into text, so it fits inside JSON
from typing import Literal                   # Literal["a", "b"] = "only these exact values are allowed"
import numpy as np                           # to turn the heatmap into an image
import json                                  # to read thresholds.json
import pandas as pd                          # to read fracture_metrics.csv
import torch                                 # PyTorch: runs the models
import timm                                  # library that has the ConvNeXt model
from PIL import Image, ImageDraw             # opens/resizes images; ImageDraw draws the box
from transformers import AutoImageProcessor, AutoModel   # library that has the DINOv2 model
from fastapi import FastAPI, File, UploadFile, HTTPException, Query   # the API library (explained below)
from fastapi.middleware.cors import CORSMiddleware            # lets the website (another address) call us
import main                                  # our main.py: Head (CNN+DINO head), label_map, device
import FractureLocator as fl                 # our FractureLocator.py: PatchHead, find_box, image_scores


# ---------- settings (CAPITALS = values you might want to change) ----------
PASSES = 5                                   # Monte Carlo dropout passes per head (5 + 5 = 10 votes)
DINO_SIZE = 448                              # same as DINO.py / DINO_patches.py
MAX_UPLOAD_MB = 15                           # refuse files bigger than this (protects the server)
HEATMAP_STRENGTH = 0.6                       # at most 60% red over a pixel, so the bone stays visible
BOX_COLOR = (255, 255, 0)                    # yellow (red, green, blue: 0-255 each)
HEAT_COLOR = (255, 0, 0)                     # red
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
# ConvNeXt's own resize + normalisation. (** = "use every setting in cnn_config as an argument")
cnn_transform = timm.data.create_transform(**cnn_config, is_training=False)

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


def draw_overlay(original, heatmap, box, show):
    # Draws the result on the ORIGINAL upload (its own size, not 1024 x 1024) and returns it as text.
    #   original = the uploaded X-ray (PIL image)
    #   heatmap  = [32, 32] numpy grid of fracture probabilities 0-1
    #   box      = {"x1": ..., "y1": ..., "x2": ..., "y2": ...} as 0-1 fractions, or None (normal)
    #   show     = "box", "heatmap" or "both"
    picture = original.convert("RGB")                           # a copy we can paint on, 3 channels
    width, height = picture.size

    if show == "heatmap" or show == "both":
        # 1. the 32 x 32 grid -> a gray image (0-255), stretched smoothly (BILINEAR) to the X-ray's size
        heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
        # 2. how strongly each pixel turns red: probability x HEATMAP_STRENGTH (so 1.0 -> 60% red)
        heat = np.asarray(heat).astype(np.float32) * HEATMAP_STRENGTH
        heat = Image.fromarray(heat.astype(np.uint8))
        # 3. Image.composite(A, B, mask): where the mask is bright take A (red), where dark keep B (the X-ray)
        red = Image.new("RGB", picture.size, HEAT_COLOR)
        picture = Image.composite(red, picture, heat)

    if (show == "box" or show == "both") and box is not None:  # no box for "normal" answers
        # fractions -> pixels of THIS image, then draw the rectangle
        corners = [box["x1"] * width, box["y1"] * height, box["x2"] * width, box["y2"] * height]
        line_width = max(2, round(min(width, height) / 200))    # thicker line on bigger images
        ImageDraw.Draw(picture).rectangle(corners, outline=BOX_COLOR, width=line_width)

    # picture -> PNG bytes in memory -> base64 text -> a "data URL" the website can use directly:
    #   <img src="data:image/png;base64,iVBORw0KG...">
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def analyse(image, show):
    # image = PIL image as uploaded, show = "box" / "heatmap" / "both".
    # Returns a dictionary (becomes the JSON answer).
    original = image                         # keep the upload as it is, to draw on at the end

    # --- step 1: same size as training ---
    image = image.convert("RGB").resize((1024, 1024))   # gray X-ray -> 3 channels, 1024 x 1024

    with torch.no_grad():                    # no training here, so no gradients (faster, less memory)
        # --- step 2: ConvNeXt vector ---
        cnn = cnn_model(cnn_transform(image).unsqueeze(0).to(device))   # [1, 1024]

        # --- step 3: DINO. ONE pass gives both the image vector and the square vectors ---
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        tokens = dino_model(pixel_values=inputs["pixel_values"]).last_hidden_state   # [1, 1 + 1024, 768]
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
        "box": None,                         # None becomes null in JSON = "no box"
        "heatmap": None,                     # filled below
        "show": show,
        "image": None,                       # filled below
        "disclaimer": "Not a diagnosis. Always have a veterinarian review the X-ray.",
    }

    # --- step 6: heatmap (ALWAYS, also for "normal") and yellow box (only for "fracture") ---
    mean_heatmap = torch.stack(heatmaps).mean(0).numpy()       # average of the 5 passes, [32, 32]
    result["heatmap"] = mean_heatmap.round(3).tolist()         # 32 rows of 32 numbers, so the website could draw it too
    if is_fracture:
        x1, y1, x2, y2 = fl.find_box(mean_heatmap)              # pixels of the 1024 x 1024 image
        result["box"] = to_fraction([x1, y1, x2, y2], 1024, 1024)

    # --- step 7: the picture the caller asked for ---
    result["image"] = draw_overlay(original, mean_heatmap, result["box"], show)
    return result



# ======================================================================================
# 3. THE API ITSELF (FastAPI)
#    app = the API object. Uvicorn (the server program) runs it: "uvicorn API:app".
#    @app.get / @app.post = "when a request comes to this address, run the function below".
#      GET  = ask for something (like opening a web page)
#      POST = send something (like uploading a file)
# ======================================================================================

app = FastAPI(title="Radiant fracture API",
              description="Upload a dog or cat limb X-ray: fracture yes/no, confidence, box, heatmap, "
                          "and the X-ray with the box and/or heatmap drawn on it.")

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
async def predict(file: UploadFile = File(...),
                  show: Literal["box", "heatmap", "both"] = Query("both")):
    # The website sends the X-ray here as a "multipart" upload (a normal file upload).
    # UploadFile = the uploaded file. File(...) = it is required (... means "must be given").
    # show = what to draw on the returned picture. It comes from the address ("query parameter"):
    #   /predict?show=box   /predict?show=heatmap   /predict?show=both   (just /predict = "both")
    #   Literal[...] makes FastAPI refuse anything else by itself
    #   (error 422, with a message saying which values are allowed).
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

    return analyse(image, show)              # FastAPI turns the dictionary into JSON automatically
