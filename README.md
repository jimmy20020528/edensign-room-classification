# edensign-room-classification

Self-hosted room-type classification + photo instance grouping for real-estate
listing photos. Runs on **CPU** (no GPU needed). Exposes one HTTP endpoint that
takes property photos and returns, per photo, the room type + occupancy, and
groups photos that show the same physical room.

- **Model:** DINOv2-base (ViT-B, frozen) embeddings → 3 linear probes
  (occupancy, furnished room-type, empty room-type) + DINOv2 patch matching for
  instance grouping.
- **No external API / VLM calls.** Everything runs locally from the bundled
  `artifacts/`.

## API

### `POST /classify-rooms`
`multipart/form-data`, field **`files`** = 1–30 image files (jpg/png).

```bash
curl -X POST http://HOST:8003/classify-rooms \
  -F "files=@room1.jpg" \
  -F "files=@room2.jpg" \
  -F "files=@room3.jpg"
```

Response `200`:
```json
{
  "photos": [
    {"index": 0, "room_type": "kitchen",  "occupancy": "furnished", "confidence": 0.91, "group_id": 1},
    {"index": 1, "room_type": "kitchen",  "occupancy": "furnished", "confidence": 0.88, "group_id": 1},
    {"index": 2, "room_type": "bedroom",  "occupancy": "empty",     "confidence": 0.79, "group_id": 2}
  ],
  "groups": [
    {"group_id": 1, "room_type": "kitchen", "occupancy": "furnished", "photo_indices": [0, 1]},
    {"group_id": 2, "room_type": "bedroom", "occupancy": "empty",     "photo_indices": [2]}
  ]
}
```
- `index` = position of the file in the request (0-based).
- `group_id` links photos of the same physical room; `groups` is the inverse view.
- Errors: `400` (0 or >30 images), `503` (models still loading).

### `GET /health`
```json
{"status": "ok", "ready": true}
```
`ready` is `false` until DINOv2 + classifiers finish loading on startup. Poll
this before sending traffic.

## Run with Docker (recommended)

```bash
docker build -t edensign-room-classification .
docker run -p 8003:8003 edensign-room-classification
# first start: model already baked in the image; ready in a few seconds
curl localhost:8003/health
```

## Run without Docker (venv)

```bash
python3 -m venv .venv && source .venv/bin/activate
# CPU-only torch (smaller than the default CUDA build)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003
```
First start downloads DINOv2-base (~350 MB) from HuggingFace into
`~/.cache/huggingface` (one time), then it is cached.

## Hardware

CPU only. Recommended **4 vCPU / 8 GB RAM** (minimum 2 vCPU / 4 GB). ~5 GB disk
(torch + transformers + DINOv2 cache). A classify call for ~30 photos takes
roughly 15–60 s depending on vCPU.

## Layout

```
app/main.py              FastAPI service (endpoints + classify/group orchestration)
scripts/group_instances.py  DINOv2 loading, feature extraction, VLAD + RANSAC grouping
artifacts/               trained classifiers + class names + VLAD vocab (loaded at startup)
```
