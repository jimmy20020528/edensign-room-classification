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

Classify 1–30 listing photos by room type and occupancy, and group photos that
show the same physical room. You send image URLs; the service downloads them
server-side and processes them.

**Request** — `Content-Type: application/json`

```json
{
  "image_urls": [
    "https://content.edensign.io/images/abc.jpg",
    "https://content.edensign.io/images/def.jpg",
    "https://content.edensign.io/images/ghi.jpg"
  ]
}
```

| field | type | required | description |
|---|---|---|---|
| `image_urls` | string[] | yes | 1–30 publicly-reachable image URLs (jpg/png), downloaded server-side |

**Response** — `200 OK`

```json
{
  "groups": [
    {
      "id": 1,
      "room_type": "kitchen",
      "occupancy": "furnished",
      "photos": [
        {"url": "https://content.edensign.io/images/abc.jpg", "room_type": "kitchen", "occupancy": "furnished", "confidence": 0.91},
        {"url": "https://content.edensign.io/images/def.jpg", "room_type": "kitchen", "occupancy": "furnished", "confidence": 0.88}
      ]
    },
    {
      "id": 2,
      "room_type": "bed",
      "occupancy": "empty",
      "photos": [
        {"url": "https://content.edensign.io/images/ghi.jpg", "room_type": "bed", "occupancy": "empty", "confidence": 0.79}
      ]
    }
  ]
}
```

`groups[]` — one object per distinct physical room:

| field | type | description |
|---|---|---|
| `id` | int | group id (1-based); photos in the same group are the same physical room |
| `room_type` | string (enum) | room type of this group (one of the values below) |
| `occupancy` | string (enum) | `"furnished"` or `"empty"` |
| `photos` | object[] | the photos that belong to this room |

`groups[].photos[]` — one object per input URL assigned to this room:

| field | type | description |
|---|---|---|
| `url` | string | the source image URL from the request (use this to match back to your input) |
| `room_type` | string (enum) | same as the group's `room_type` |
| `occupancy` | string (enum) | `"furnished"` or `"empty"` |
| `confidence` | float | 0.0–1.0, model confidence of `room_type` |

**`room_type` enum** (13 values, aligned with the backend `RoomType`): `living`,
`bed`, `kitchen`, `dining`, `bathroom`, `home_office`, `outdoor`, `kids_room`,
`hallway`, `theatre`, `living_bedroom`, `living_dining`, `balcony`.

**Errors** — body is always `{"detail": "<message>"}`:

| status | when | example body |
|---|---|---|
| `400` | `image_urls` empty | `{"detail": "At least 1 image_url required"}` |
| `400` | more than 30 urls | `{"detail": "Max 30 images"}` |
| `502` | a URL could not be downloaded | `{"detail": "Failed to download image 2: https://... (...)"}` |
| `503` | service still loading models | `{"detail": "Models not loaded"}` |

### `GET /health`

**Response** — `200 OK`

```json
{"status": "ok", "ready": true}
```

`ready` is `false` until DINOv2 + the classifiers finish loading on startup.
Poll this and wait for `ready: true` before sending classify traffic.

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
