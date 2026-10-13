# How to make an API (and what we did for Radiant)

## 1. What is an API?

An **API** (Application Programming Interface) is a program that **waits for requests from other programs and
sends answers back**. Your website can't run Python or PyTorch: GitHub Pages only serves static HTML/JS files.
So the model lives in a separate program, the API. The website sends it an X-ray and gets an answer back:

```
Website (browser, JavaScript)                    API (Python, API.py)
  user picks an X-ray  ── POST /predict + image ──►  runs ConvNeXt + DINO + heads + YOLO
  shows answer + boxes ◄──── JSON answer ─────────  {"answer": "fracture", "confidence": 1.0, ...}
```

**JSON** is plain text that both JavaScript and Python understand: `{"name": value, ...}`.

### API vs API key (they're different things!)
- **API** = the program that answers requests. **We built one: `API.py`.**
- **API key** = a secret password some APIs require, sent with every request, so only allowed users get answers.

We did **not** add an API key, on purpose. Your website is public JavaScript: anyone can open the browser's dev tools
and read a key hidden in it, so it wouldn't protect anything. Real protection for a public site comes from
the server instead: limiting the upload size (done), CORS (done, explained below), and later rate limits on the host.
API keys make sense when **another server** (not a browser) calls your API, because a server can keep a secret.

---

## 2. The words you need

| Word | Meaning | In our API |
|---|---|---|
| **endpoint** | one address of the API that does one job | `/` (health check), `/predict` (analyse an X-ray) |
| **GET** | a request that asks for something | opening `http://127.0.0.1:8000/` |
| **POST** | a request that sends something | uploading the X-ray to `/predict` |
| **request / response** | what the caller sends / what the API sends back | image in, JSON out |
| **status code** | number saying how it went | 200 OK, 400 bad file, 413 file too big |
| **FastAPI** | Python library to write APIs with very little code | `app = FastAPI()` |
| **uvicorn** | the server program that runs a FastAPI app and listens on a port | `uvicorn API:app` |
| **port** | which "door" of the computer the API listens on | 8000 |
| **CORS** | browser rule: a page may only call another address if that address allows it | we allow your github.io site |
| **multipart / UploadFile** | the normal way browsers upload files | `file: UploadFile` |

---

## 3. How to make an API yourself, step by step

**Step 1: install.**
```bash
pip install fastapi uvicorn python-multipart
```
`python-multipart` is needed to receive uploaded files.

**Step 2: the smallest possible API** (save it as `hello.py`).
```python
from fastapi import FastAPI

app = FastAPI()                      # the API object

@app.get("/")                        # when someone GETs the address "/" ...
def home():                          # ... run this function ...
    return {"message": "hello"}      # ... and send this back as JSON
```

**Step 3: run it.**
```bash
python -m uvicorn hello:app --reload
```
`hello:app` means "the variable `app` in the file `hello.py`". `--reload` restarts it when you save the file.
Open http://127.0.0.1:8000/ and you see `{"message":"hello"}`.

**Step 4: the free test page.** Open http://127.0.0.1:8000/docs. FastAPI builds a page where you can try every
endpoint, including uploading files, without writing any website code.

**Step 5: receive data.** An endpoint that takes an uploaded file:
```python
from fastapi import FastAPI, File, UploadFile

@app.post("/size")
async def size(file: UploadFile = File(...)):
    data = await file.read()                 # the file's bytes
    return {"bytes": len(data)}
```

**Step 6: send errors properly.**
```python
from fastapi import HTTPException
raise HTTPException(status_code=400, detail="That is not an image")
```

**Step 7: allow your website (CORS).**
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["https://realpkim-gif.github.io"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
```

**Step 8: load models ONCE, at the top of the file**, not inside the endpoint.
Loading takes seconds; doing it per request would make every answer slow.

---

## 4. What we did in this project (`API.py`)

`API.py` has three parts:

**Part 1: load everything once.** When uvicorn starts the API, it loads:
- ConvNeXt and DINOv2 (the frozen backbones)
- the trained **CNN+DINO head** (`head.pt`) and **patch head** (`fracture_head.pt`)
- the two **thresholds** the heads vote with (`thresholds.json`, `fracture_metrics.csv`), both chosen on val to catch 90% of fractures
- the **best YOLO epoch**: `runs/detect/fracture_type_yolo/weights/best.pt` (`best.pt` = the best epoch on val; `last.pt` = simply the last one)

**Part 2: `analyse(image)`, the pipeline for one X-ray.** It uses the same steps as training, so the models get
the same kind of input they learned on:
1. Resize to 1024 x 1024 (the training X-rays were that size).
2. ConvNeXt makes the cnn vector [1, 1024]. **One** DINO pass gives the image vector [1, 1536] **and** the square vectors [1, 32, 32, 768].
3. **Monte Carlo dropout:** the CNN+DINO head runs 5 times and the patch head 5 times, with dropout on, giving **10 votes**.
4. Majority of the votes = **answer**; the share that agrees = **confidence**.
5. If it's a fracture: the average heatmap gives the **rough (yellow) box**. Then the X-ray is cropped to that box + 15%,
   and **YOLO** finds the **precise box** and the **fracture type** inside it.

**Part 3: the endpoints.**
- `GET /`: `{"status": "ok", "device": "cuda"}`, to check the API is alive.
- `POST /predict`: receives the X-ray, refuses files over 15 MB (413) or files that aren't images (400),
  runs `analyse()`, and returns:

```json
{
  "answer": "fracture",
  "confidence": 1.0,
  "fracture_votes": "10 of 10",
  "rough_box":   {"x1": 0.5312, "y1": 0.3438, "x2": 0.875,  "y2": 0.625},
  "precise_box": {"x1": 0.7744, "y1": 0.5574, "x2": 0.9259, "y2": 0.6434},
  "fracture_type": "epiphysis",
  "type_confidence": 0.54,
  "disclaimer": "Not a diagnosis. Always have a veterinarian review the X-ray."
}
```

Boxes are **fractions (0-1)** of the image width/height, so the website can draw them at any display size:
`left = x1 * shownWidth`, `top = y1 * shownHeight`, and so on. For a "normal" answer, the boxes and type are `null`.

**Tested:** a fracture X-ray gave 10 of 10 votes and the same box as vote.py; a normal one gave 0 of 10;
a text file gave a 400 error.

---

## 5. Calling it from the website (JavaScript, for the Analyze page)

```javascript
const API_URL = "http://127.0.0.1:8000";        // later: your hosted API address

async function analyse(file) {                   // file = from <input type="file">
  const form = new FormData();
  form.append("file", file);                     // the name must be "file", like in API.py
  const response = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
  if (!response.ok) throw new Error((await response.json()).detail);
  return await response.json();                  // {answer, confidence, rough_box, ...}
}
```

---

## 6. Running it on your laptop vs on the internet

**On your laptop (for testing):**
```bash
python -m uvicorn API:app --host 127.0.0.1 --port 8000
```
Run it from this project folder: it needs `head.pt`, `fracture_head.pt`, `thresholds.json`, `fracture_metrics.csv`,
`PreprocessedData/label_map.json` and the YOLO `best.pt`.

**On the internet (so the real website can use it):** the API must run on a server, e.g. Hugging Face Spaces or Render.
- They give you an `https://...` address. Put it in `API_URL` in the website, and keep the github.io address in `ALLOWED_WEBSITES`.
- Free servers usually have **no GPU**. The code then runs on CPU automatically (`device = "cpu"`), which is slower
  (several seconds per X-ray) but works.
- The model files (`.pt`) are big and are not in git. Upload them to the server, or to a private Hugging Face model repo.
- **Licenses:** Ultralytics YOLO is AGPL-3.0, so a public service using it must publish its source or buy a license.
