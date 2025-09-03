## OpenCLIP as a Service

Minimal FastAPI service that serves text embeddings from OpenCLIP.

### What you get
- **Endpoint**: `POST /embed` → returns embeddings for input texts
- **Model**: `openai/clip-vit-base-patch32` via `open-clip-torch`
- **Device**: Auto-selects CUDA if available, else CPU
- **Dimension**: 512-d embeddings; optional L2 normalization

---

## Quickstart

### 1) Install dependencies
You can use either `uv` (fast) or `pip`.

```bash
# Using uv (recommended)
uv pip install fastapi uvicorn open-clip-torch torch

# Or using pip
pip install fastapi uvicorn open-clip-torch torch
```

If you plan to expose the service over the internet from a notebook, also install:

```bash
pip install pyngrok
```

### 2) Start the service (from the notebook)
This repo contains `openclip_service.ipynb` with a ready-to-run FastAPI app.

Steps:
1. Open `openclip_service.ipynb`.
2. Run the install cell if needed.
3. Run the cell that defines the FastAPI `app` and the `/embed` route.
4. Start the server: Use the ngrok cell to run uvicorn in a background thread and expose a public URL.

Local server default: `http://localhost:8000`

Tip: To run outside notebooks, you can convert the notebook to a script:

```bash
jupyter nbconvert --to script openclip_service.ipynb
python openclip_service.py
```

Or copy the FastAPI snippet from the notebook into your own `app.py` and run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3) Optional: expose via ngrok
The notebook includes a cell that starts uvicorn in a background thread and creates an ngrok tunnel.

You’ll be prompted for your ngrok authtoken (get one from `https://dashboard.ngrok.com/get-started/your-authtoken`). The cell prints a public URL such as `https://xyz.ngrok-free.app`.

Use that base URL in requests (see examples below).

---

## API

### Health
```http
GET /
```
Response:
```json
{ "status": "ok" }
```

### Embed
```http
POST /embed
Content-Type: application/json
```

Request body:
```json
{
  "texts": ["a photo of a dog", "a red car"],
  "normalize": true
}
```

- **texts**: array of strings (required)
- **normalize**: boolean (optional, default `true`). If `true`, L2-normalizes embeddings.

Response body:
```json
{
  "embeddings": [[0.12, 0.01, ...], [0.03, -0.07, ...]]
}
```

- Shape: `[num_texts, 512]`
- Dtype: float32 (JSON numbers)

Auth: No authentication is enforced by default.

---

## Examples

Assume the service is running at `http://localhost:8000`.

### curl (PowerShell)
```powershell
curl -Method POST `
  -Uri http://localhost:8000/embed `
  -ContentType 'application/json' `
  -Body '{"texts":["a photo of a dog","a red car"],"normalize":true}'
```

### curl (bash)
```bash
curl -s -X POST http://localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["a photo of a dog","a red car"],"normalize":true}'
```

### Python (requests)
```python
import requests

url = "http://localhost:8000/embed"
payload = {"texts": ["a photo of a dog", "a red car"], "normalize": True}
r = requests.post(url, json=payload, timeout=15)
r.raise_for_status()
embeddings = r.json()["embeddings"]  # List[List[float]], shape [N, 512]
print(len(embeddings), len(embeddings[0]))
```

### Provided test script
`test_embed_api.py` exercises the API and prints the returned shape.

```bash
python test_embed_api.py --url http://localhost:8000 "a photo of a dog" "a red car"
```

If you exposed the service via ngrok, pass the public URL:

```bash
python test_embed_api.py --url https://YOUR-TUNNEL.ngrok-free.app "a photo of a dog" "a red car"
```

You can also pass an Authorization header (the server ignores it by default, but it’s useful if you add auth later):

```bash
python test_embed_api.py --url http://localhost:8000 --key YOUR_KEY "hello world"
```

---

## Use a different OpenCLIP model
You can replace the model with any OpenCLIP-supported checkpoint.

Option A — Hugging Face Hub repo id (current approach):

1. Edit `MODEL_ID` in `openclip_service.ipynb`.
2. Set it to another HF repo, prefixed with `hf-hub:`. No other code changes are required when using Hub IDs.

Option B — Built-in OpenCLIP names and weights:

Replace the creation calls in the notebook like this:

```python
model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=DEVICE)
tokenizer = open_clip.get_tokenizer(model_name)
```

Notes:
- Embedding dimensionality depends on the chosen model (e.g., 512 for `ViT-B/32`, 768 for `ViT-L/14`, etc.). Client code should not assume a fixed size.
- If you change the model, restart the kernel/server so the new weights are loaded.


