# How to make an API (and what we did for Radiant)

## 1. What is an API?

An **API** (Application Programming Interface) is a program that **waits for requests from other programs and
sends answers back**. Your website can't run Python or PyTorch: GitHub Pages only serves static HTML/JS files.
So the model lives in a separate program, the API. The website sends it an X-ray and gets an answer back:

```
Website (browser, JavaScript)                    API (Python, API.py)
  user picks an X-ray  ── POST /predict + image ──►  runs ConvNeXt + DINO + the two heads
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

**Part 2: `analyse(image)`, the pipeline for one X-ray.** It uses the same steps as training, so the models get
the same kind of input they learned on:
1. Resize to 1024 x 1024 (the training X-rays were that size).
2. ConvNeXt makes the cnn vector [1, 1024]. **One** DINO pass gives the image vector [1, 1536] **and** the square vectors [1, 32, 32, 768].
3. **Monte Carlo dropout:** the CNN+DINO head runs 5 times and the patch head 5 times, with dropout on, giving **10 votes**.
4. Majority of the votes = **answer**; the share that agrees = **confidence**.
5. The average of the patch head's 5 heatmaps = the **heatmap** (always). If it's a fracture, the heatmap also gives
   the **yellow box**.
6. `draw_overlay()` paints the heatmap and/or the box on the **original upload**, depending on `show` (next section),
   and turns the picture into text (base64) so it fits in the JSON answer.

**Part 3: the endpoints.**
- `GET /`: `{"status": "ok", "device": "cuda"}`, to check the API is alive.
- `POST /predict?show=...`: receives the X-ray, refuses files over 15 MB (413) or files that aren't images (400),
  runs `analyse()`, and returns JSON.

### The `show` option: box, heatmap, or both

`show` is a **query parameter**: an extra setting written in the address after a `?`.

| Address | Picture you get back |
|---|---|
| `/predict?show=box` | the X-ray with **only the yellow box** |
| `/predict?show=heatmap` | the X-ray with **only the red heatmap** |
| `/predict?show=both` (or just `/predict`) | **both** |
| `/predict?show=banana` | error **422**: FastAPI itself answers that only `box`, `heatmap` or `both` are allowed |

That check costs one line, thanks to `Literal["box", "heatmap", "both"]` in `predict()`.

The answer always contains everything; `show` only changes the **picture**:

```json
{
  "answer": "fracture",
  "confidence": 1.0,
  "fracture_votes": "10 of 10",
  "box": {"x1": 0.5312, "y1": 0.3438, "x2": 0.875, "y2": 0.625},
  "heatmap": [[0.001, 0.002, ...], ...],
  "show": "both",
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "disclaimer": "Not a diagnosis. Always have a veterinarian review the X-ray."
}
```

- `box`: **fractions (0-1)** of the image width/height, so it fits any display size. `null` when the answer is "normal".
- `heatmap`: 32 rows of 32 fracture probabilities (0-1), one per square of the image. Always there, also for "normal".
- `image`: the uploaded X-ray **at its own size**, with the chosen drawing on it, as a **data URL**
  (a whole PNG picture written as text). Put it straight into an `<img>` tag.
- Monte Carlo dropout is random, so sending the same image twice can move the box edge by one square (32 px).

**Tested:** `box`, `heatmap` and `both` each returned the right picture; `banana` gave a 422 error;
a normal X-ray gave 0 of 10 votes; a text file gave a 400 error.

---

## 5. Calling it from the website (JavaScript, for the Analyze page)

```javascript
const API_URL = "http://127.0.0.1:8000";        // later: your hosted API address

async function analyse(file, show = "both") {   // file = from <input type="file">; show = "box" | "heatmap" | "both"
  const form = new FormData();
  form.append("file", file);                     // the name must be "file", like in API.py
  const response = await fetch(`${API_URL}/predict?show=${show}`, { method: "POST", body: form });
  if (!response.ok) throw new Error(JSON.stringify((await response.json()).detail));
  return await response.json();                  // {answer, confidence, box, heatmap, image, ...}
}

// show the picture:  document.querySelector("#result-image").src = (await analyse(file, "box")).image;
```

---

## 6. Running it on your laptop vs on the internet

**On your laptop (for testing):**
```bash
python -m uvicorn API:app --host 127.0.0.1 --port 8000
```
Run it from this project folder: it needs `head.pt`, `fracture_head.pt`, `thresholds.json`, `fracture_metrics.csv`,
and `PreprocessedData/label_map.json`.

**On the internet (so the real website can use it):** the API must run on a server, e.g. Hugging Face Spaces or Render.
- They give you an `https://...` address. Put it in `API_URL` in the website, and keep the github.io address in `ALLOWED_WEBSITES`.
- Free servers usually have **no GPU**. The code then runs on CPU automatically (`device = "cpu"`), which is slower
  (several seconds per X-ray) but works.
- The model files (`.pt`) are big and are not in git. Upload them to the server, or to a private Hugging Face model repo.
