"""cv-models FastAPI service — room classification + instance grouping.

Startup: loads DINOv2 + three sklearn classifiers from artifacts/.
Endpoint: POST /classify-rooms — accepts 1-30 images, returns per-photo
room_type/occupancy/confidence/group_id and groups list.
"""
import json
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ARTIFACTS = ROOT / "artifacts"

# CLS-based thresholds (fallback when VLAD vocab not available).
# Empirical: true same-room cosine ~0.74, different-room-same-listing ~0.61.
# t_cls_screen=0.60 intentionally loose — stricter would miss valid pairs.
# t_cls_confirm=0.72 is the actual quality gate.
_CLS_THRESHOLDS = {
    "furnished": {"t_screen": 0.60, "t_patch": 30, "t_confirm": 0.72},
    "empty":     {"t_screen": 0.60, "t_patch": 15, "t_confirm": 0.72},
}

# VLAD-based thresholds (used when artifacts/vlad_vocab.npy exists).
# VLAD screen + RANSAC geometric verification.
# t_screen: loose VLAD pre-filter to limit RANSAC calls.
# t_inliers: min geometrically consistent patch pairs to confirm same room.
_VLAD_THRESHOLDS = {
    "furnished": {"t_screen": 0.15, "t_inliers": 10},
    "empty":     {"t_screen": 0.10, "t_inliers": 6},
}

_state: dict[str, Any] = {}


def _load_artifacts() -> dict[str, Any]:
    """Load all models and class name maps. Returns state dict."""
    # Lazy import: torch + transformers are heavy; defer until lifespan startup
    # so `import app.main` in tests is fast and doesn't require DINOv2.
    from group_instances import load_dinov2, load_vlad_vocab  # deferred heavy imports
    processor, model, device = load_dinov2()
    vlad_vocab = load_vlad_vocab()   # None if vocab not built yet
    return {
        "ready": True,
        "processor": processor,
        "model": model,
        "device": device,
        "vlad_vocab": vlad_vocab,
        "occ_clf": joblib.load(ARTIFACTS / "classifier_occupancy.pkl"),
        "clf_furnished": joblib.load(ARTIFACTS / "classifier_furnished.pkl"),
        "clf_empty": joblib.load(ARTIFACTS / "classifier_empty.pkl"),
        "class_names_furnished": json.loads(
            (ARTIFACTS / "class_names_furnished.json").read_text()
        ),
        "class_names_empty": json.loads(
            (ARTIFACTS / "class_names_empty.json").read_text()
        ),
        "class_names_occupancy": json.loads(
            (ARTIFACTS / "class_names_occupancy.json").read_text()
        ),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state.update(_load_artifacts())
    except Exception as e:
        print(f"[cv-models] FATAL: could not load artifacts: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        _state["ready"] = False
    yield
    _state.clear()


app = FastAPI(title="cv-models", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ready": _state.get("ready", False)}


@app.post("/classify-rooms")
async def classify_rooms(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not _state.get("ready"):
        raise HTTPException(503, "Models not loaded")
    if len(files) == 0:
        raise HTTPException(400, "At least 1 image required")
    if len(files) > 30:
        raise HTTPException(400, "Max 30 images")

    tmp_paths: list[Path] = []
    try:
        for f in files:
            content = await f.read()
            suffix = Path(f.filename or "img.jpg").suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(content)
            tmp.close()
            tmp_paths.append(Path(tmp.name))
        return _classify_and_group(tmp_paths)
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


def _classify_and_group(image_paths: list[Path]) -> dict[str, Any]:
    # Lazy import: deferred heavy imports, no-op cache hit at request time.
    from group_instances import (  # deferred import
        count_ransac_inliers,
        encode_vlad_batch,
        extract_features,
        find_connected_components,
    )

    processor = _state["processor"]
    model = _state["model"]
    device = _state["device"]
    vlad_vocab = _state.get("vlad_vocab")
    N = len(image_paths)

    # Single DINOv2 forward pass for all images (CLS + patch features)
    cls_arr, patches_arr = extract_features(image_paths, processor, model, device)

    # Step 1: Occupancy classification (binary: furnished vs empty)
    occ_class_names = _state["class_names_occupancy"]
    raw_preds = _state["occ_clf"].predict(cls_arr)
    occ_preds: list[str] = [occ_class_names[str(int(p))] for p in raw_preds]

    # Step 2: Room type classification, routed by occupancy
    room_types: list[str] = []
    confidences: list[float] = []
    for i in range(N):
        if occ_preds[i] == "furnished":
            clf = _state["clf_furnished"]
            names = _state["class_names_furnished"]
        else:
            clf = _state["clf_empty"]
            names = _state["class_names_empty"]
        probs = clf.predict_proba([cls_arr[i]])[0]
        best = int(np.argmax(probs))
        room_types.append(names.get(str(clf.classes_[best]), f"room_{clf.classes_[best]}"))
        confidences.append(float(probs[best]))

    # Step 3: Instance grouping — run per (room_type, occupancy) bucket.
    # Choose descriptor and thresholds based on VLAD availability.
    if vlad_vocab is not None:
        global_feat = encode_vlad_batch(patches_arr, vlad_vocab)  # (N, K*768)
        thresholds = _VLAD_THRESHOLDS
    else:
        global_feat = cls_arr
        thresholds = _CLS_THRESHOLDS

    group_id_map: dict[int, int] = {}
    next_gid = 1

    bucket_keys = set(zip(room_types, occ_preds))
    for rt_label, occ_label in sorted(bucket_keys):
        bucket = [i for i in range(N)
                  if room_types[i] == rt_label and occ_preds[i] == occ_label]
        if not bucket:
            continue
        if occ_label not in thresholds:
            print(f"[cv-models] WARNING: unknown occupancy label {occ_label!r}, using furnished thresholds", file=sys.stderr)
        t = thresholds.get(occ_label, thresholds["furnished"])
        b_feat = global_feat[bucket]
        b_patches = patches_arr[bucket]

        # Step A: VLAD cosine screen — fast candidate filter
        sim = b_feat @ b_feat.T
        candidates = [
            (i, j)
            for i in range(len(bucket))
            for j in range(i + 1, len(bucket))
            if sim[i, j] > t["t_screen"]
        ]

        # Step B: RANSAC geometric verification on DINOv2 patch correspondences
        confirmed = [
            (i, j) for i, j in candidates
            if count_ransac_inliers(b_patches[i], b_patches[j]) >= t["t_inliers"]
        ]

        for component in find_connected_components(len(bucket), confirmed):
            gid = next_gid
            next_gid += 1
            for local_idx in component:
                group_id_map[bucket[local_idx]] = gid

    # Every node is already in group_id_map: find_connected_components returns
    # ALL nodes (including singletons as single-element components), so the
    # loop above covers every photo index. No ungrouped fallback is needed.

    # Build response
    photos = [
        {
            "index": i,
            "room_type": room_types[i],
            "occupancy": occ_preds[i],
            "confidence": confidences[i],
            "group_id": group_id_map[i],
        }
        for i in range(N)
    ]

    groups_by_id: dict[int, dict] = {}
    for p in photos:
        gid = p["group_id"]
        if gid not in groups_by_id:
            groups_by_id[gid] = {
                "group_id": gid,
                "room_type": p["room_type"],
                "occupancy": p["occupancy"],
                "photo_indices": [],
            }
        groups_by_id[gid]["photo_indices"].append(p["index"])

    return {
        "photos": photos,
        "groups": sorted(groups_by_id.values(), key=lambda g: g["group_id"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
