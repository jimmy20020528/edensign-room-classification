"""Task 2: Room Instance Grouping

Given a set of photos (from one property), group photos that show the same
physical room — even across different camera angles.

Algorithm:
  Step A (coarse, fast): pairwise global descriptor cosine similarity.
                         If vlad_vocab.npy exists: VLAD over DINOv2 patches
                         (much more instance-discriminative than CLS token).
                         Fallback: DINOv2 CLS cosine similarity.
  Step B (fine, slow):   for candidate pairs, count mutual best patch matches
                         (DINOv2 patch features, 256 per image).
                         Pairs with mutual_count >= T_PATCH are confirmed.
  Step C:                strict global descriptor re-check + connected components.

No training. Uses frozen DINOv2-base, same backbone as Task 1.
Build VLAD vocab first: python scripts/build_vlad_vocab.py

Usage:
  python scripts/group_instances.py --folder <path> [--t-cls 0.70] [--t-patch 30]
  python scripts/group_instances.py --files img1.jpg img2.jpg img3.jpg
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
VLAD_VOCAB_PATH = ROOT / "artifacts" / "vlad_vocab.npy"

DEFAULT_T_CLS = 0.70    # CLS fallback: true same-room ~0.74, diff-room ~0.61
DEFAULT_T_PATCH = 30    # mutual best match count threshold for confirmation


def load_dinov2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    return processor, model, device


def load_vlad_vocab():
    """Load VLAD vocabulary if available. Returns (K, 768) array or None."""
    if VLAD_VOCAB_PATH.exists():
        vocab = np.load(VLAD_VOCAB_PATH)
        print(f"  VLAD vocab loaded: {vocab.shape} ({vocab.shape[0]} clusters)",
              file=sys.stderr)
        return vocab
    return None


def encode_vlad_batch(patches_arr, vocab):
    """Encode patch features into VLAD descriptors.

    patches_arr: (N, P, 768) L2-normalized patch features
    vocab:       (K, 768) L2-normalized cluster centroids
    Returns:     (N, K*768) L2-normalized VLAD descriptors
    """
    N, P, D = patches_arr.shape
    K = vocab.shape[0]
    vlads = np.zeros((N, K * D), dtype=np.float32)

    for n in range(N):
        patches = patches_arr[n]          # (P, D)
        sim = patches @ vocab.T           # (P, K)
        assignments = sim.argmax(axis=1)  # (P,)

        vlad = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                vlad[k] = (patches[mask] - vocab[k]).sum(axis=0)
                nk = np.linalg.norm(vlad[k])
                if nk > 1e-6:
                    vlad[k] /= nk          # intra-normalization

        flat = vlad.flatten()
        n_flat = np.linalg.norm(flat)
        if n_flat > 1e-6:
            flat /= n_flat                 # global L2-normalize
        vlads[n] = flat

    return vlads


@torch.no_grad()
def extract_features(image_paths, processor, model, device, batch_size=8):
    """Returns (cls_array, patches_array) where:
        cls_array: (N, 768) L2-normalized
        patches_array: (N, 256, 768) L2-normalized per patch
    """
    N = len(image_paths)
    cls_arr = np.zeros((N, 768), dtype=np.float32)
    patches_arr = np.zeros((N, 256, 768), dtype=np.float32)

    for start in range(0, N, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  WARN: cannot read {p}: {e}", file=sys.stderr)
                batch_imgs.append(Image.new("RGB", (224, 224), color=(128, 128, 128)))

        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        out = model(**inputs)
        hidden = out.last_hidden_state   # (B, 257, 768)

        cls = hidden[:, 0]               # (B, 768)
        norms = cls.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cls = cls / norms
        patches = hidden[:, 1:]          # (B, 256, 768)
        patch_norms = patches.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        patches = patches / patch_norms

        cls_arr[start:start + len(batch_paths)] = cls.cpu().numpy()
        patches_arr[start:start + len(batch_paths)] = patches.cpu().numpy()

    return cls_arr, patches_arr


def count_mutual_best_matches(patches_a, patches_b):
    """Count patches that mutually pick each other as best match.

    patches_a, patches_b: (P, 768) each, L2-normalized.
    Returns: int count in [0, P].
    """
    sim = patches_a @ patches_b.T        # (P, P)
    a_to_b = sim.argmax(axis=1)          # for each A patch, best B patch index
    b_to_a = sim.argmax(axis=0)          # for each B patch, best A patch index
    # Mutual: A[i] → B[j] AND B[j] → A[i], i.e. b_to_a[a_to_b[i]] == i
    return int(np.sum(b_to_a[a_to_b] == np.arange(len(a_to_b))))


def count_ransac_inliers(patches_a, patches_b, reproj_thresh=16.0, n_iter=100):
    """Geometric verification via RANSAC affine fit on DINOv2 patch correspondences.

    DINOv2-base: 16×16 patch grid, each patch 14×14 px at 224×224 input.
    Finds mutual best-match patch pairs, fits affine transform with RANSAC,
    returns number of geometrically consistent inliers. Pure numpy, no cv2.

    patches_a, patches_b: (256, 768) L2-normalized patch features.
    Returns: int inlier count (0 if fewer than 4 mutual matches).
    """
    sim = patches_a @ patches_b.T            # (256, 256)
    a_to_b = sim.argmax(axis=1)              # (256,)
    b_to_a = sim.argmax(axis=0)              # (256,)
    mutual_idx = np.where(b_to_a[a_to_b] == np.arange(256))[0]

    if len(mutual_idx) < 4:
        return 0

    # DINOv2-base patch grid: 16 cols × 16 rows, 14px per patch
    rows, cols = np.divmod(mutual_idx.astype(int), 16)
    pts_a = np.column_stack([cols * 14 + 7, rows * 14 + 7]).astype(np.float32)

    b_idx = a_to_b[mutual_idx].astype(int)
    b_rows, b_cols = np.divmod(b_idx, 16)
    pts_b = np.column_stack([b_cols * 14 + 7, b_rows * 14 + 7]).astype(np.float32)

    n = len(mutual_idx)
    best_inliers = 0
    rng = np.random.default_rng(0)

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        A_mat = np.column_stack([pts_a[idx], np.ones(3)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, pts_b[idx], rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = np.column_stack([pts_a, np.ones(n)]) @ coeffs
        err = np.linalg.norm(pred - pts_b, axis=1)
        inliers = int((err < reproj_thresh).sum())
        if inliers > best_inliers:
            best_inliers = inliers

    return best_inliers


def find_connected_components(n_nodes, edges):
    """Union-find: edges is list of (i, j) pairs. Returns list of components."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in edges:
        union(i, j)

    groups = {}
    for i in range(n_nodes):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return list(groups.values())


def group_instances(image_paths, t_cls=DEFAULT_T_CLS, t_patch=DEFAULT_T_PATCH,
                    processor=None, model=None, device=None, verbose=True):
    """Main entry point.

    Returns: list of lists of int indices into image_paths.
             e.g. [[0, 3, 4], [1], [2, 5]]
    """
    if processor is None:
        processor, model, device = load_dinov2()

    N = len(image_paths)
    if verbose:
        print(f"  Extracting DINOv2 features for {N} images...", file=sys.stderr)
    t0 = time.time()
    cls_arr, patches_arr = extract_features(image_paths, processor, model, device)
    if verbose:
        print(f"  Features extracted in {time.time() - t0:.1f}s", file=sys.stderr)

    # Choose global descriptor: VLAD (preferred) or CLS (fallback)
    vlad_vocab = load_vlad_vocab()
    if vlad_vocab is not None:
        global_feat = encode_vlad_batch(patches_arr, vlad_vocab)  # (N, K*768)
        # VLAD similarity regime is different from CLS — use adjusted threshold.
        # VLAD same-room ~0.55-0.75, different-room-same-type ~0.10-0.35.
        # Screen at 0.20 (very loose, catches all candidates cheaply),
        # confirm at 0.45 (separates same-room from false merges).
        # Tune these on your specific data using evaluate_grouping.py.
        t_screen = 0.20
        t_confirm = 0.45
        descriptor = "VLAD"
    else:
        global_feat = cls_arr
        t_screen = t_cls
        t_confirm = 0.72   # mirrors app/main.py _THRESHOLDS t_cls_confirm
        descriptor = "CLS"

    # Step A: global descriptor pairwise similarity screen
    sim = global_feat @ global_feat.T    # (N, N)
    candidate_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if sim[i, j] > t_screen:
                candidate_pairs.append((i, j, float(sim[i, j])))

    if verbose:
        print(f"  Step A: {len(candidate_pairs)} candidate pairs "
              f"({descriptor} sim > {t_screen})", file=sys.stderr)

    # Step B: mutual best patch matches for candidates
    confirmed_edges = []
    for i, j, sim_val in candidate_pairs:
        count = count_mutual_best_matches(patches_arr[i], patches_arr[j])
        passes_patch = count >= t_patch
        passes_confirm = sim_val > t_confirm
        if passes_patch and passes_confirm:
            confirmed_edges.append((i, j))
        if verbose:
            ok = "OK" if (passes_patch and passes_confirm) else "no"
            print(f"    pair ({i:2d},{j:2d})  patches={count:3d}  "
                  f"{descriptor}={sim_val:.3f}  {ok}", file=sys.stderr)

    if verbose:
        print(f"  Step B+C: {len(confirmed_edges)} confirmed same-room edges",
              file=sys.stderr)

    # Step C: connected components
    groups = find_connected_components(N, confirmed_edges)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder of images to group")
    parser.add_argument("--files", nargs="+", help="Specific image paths")
    parser.add_argument("--t-cls", type=float, default=DEFAULT_T_CLS)
    parser.add_argument("--t-patch", type=int, default=DEFAULT_T_PATCH)
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
        image_paths = sorted([p for p in folder.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    elif args.files:
        image_paths = [Path(p) for p in args.files]
    else:
        print("Usage: --folder <path> OR --files <files>...", file=sys.stderr)
        return

    if not image_paths:
        print("No images found", file=sys.stderr)
        return

    groups = group_instances(image_paths, t_cls=args.t_cls, t_patch=args.t_patch,
                             verbose=not args.json)

    if args.json:
        out = {f"group_{i+1}": [str(image_paths[idx]) for idx in g]
               for i, g in enumerate(groups)}
        print(json.dumps(out, indent=2))
    else:
        print(f"\n=== {len(groups)} groups found ===")
        for i, g in enumerate(groups):
            print(f"\nGroup {i+1} ({len(g)} photos):")
            for idx in g:
                print(f"  {image_paths[idx].name}")


if __name__ == "__main__":
    main()
