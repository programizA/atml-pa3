"""Fetch the selected COCO images and cache their frozen SigLIP patch features.

    python -m task2.features --data results/task2/data --cache /tmp/pa3_feats

For each image: deterministic SigLIP preprocessing (resize to 224x224, scale to [-1, 1]), one pass
through the frozen vision tower, keep `last_hidden_state`: the 196 final patch features in their
14x14 row-major order (768 numbers each). In Hugging Face's SigLIP that tensor has already been
through the tower's final LayerNorm (`post_layernorm`) exactly once; we verify this on the first
batch rather than assume it. Features are stored in FP16 in one memory-mapped array, and the
vision tower is deleted afterwards so training never holds it in GPU memory.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import torch

from common.utils import Timer, mb, peak_memory_mb, reset_peak_memory, write_json
from task2 import config as C


def image_path(root, k):
    split, iid = k.split("/")
    return Path(root) / split / f"COCO_{split}_{int(iid):012d}.jpg"


def fetch_images(keys, img_root, workers=32):
    """Download any missing images from the official COCO host (or use an existing local copy)."""
    import urllib.request

    missing = [k for k in keys if not image_path(img_root, k).exists()]
    t0 = time.perf_counter()

    def get(k):
        split, iid = k.split("/")
        dest = image_path(img_root, k)
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                tmp = dest.with_suffix(".part")
                urllib.request.urlretrieve(C.COCO_IMG_URL.format(split=split, id=int(iid)), tmp)
                tmp.rename(dest)
                return None
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(1 + attempt)
        return f"{k}: {err}"

    errors = []
    if missing:
        print(f"downloading {len(missing)} images ...", flush=True)
        with cf.ThreadPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(get, missing)):
                if r:
                    errors.append(r)
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{len(missing)}", flush=True)
    return {"n_images": len(keys), "downloaded": len(missing) - len(errors), "errors": errors[:20],
            "n_errors": len(errors), "seconds": time.perf_counter() - t0}


@torch.no_grad()
def cache_features(keys, img_root, cache_dir, vision_id=C.VISION_ID, batch=64):
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(vision_id, use_fast=False)   # PIL path: deterministic
    tower = SiglipVisionModel.from_pretrained(
        vision_id, dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    feats = np.lib.format.open_memmap(cache_dir / "features.npy", mode="w+", dtype=np.float16,
                                      shape=(len(keys), C.N_PATCH, C.D_VIS))
    ln_check = None
    for i in range(0, len(keys), batch):
        ims = [Image.open(image_path(img_root, k)).convert("RGB") for k in keys[i:i + batch]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, tower.dtype)
        out = tower(pixel_values=px, output_hidden_states=ln_check is None)
        h = out.last_hidden_state
        assert h.shape[1:] == (C.N_PATCH, C.D_VIS), h.shape
        if ln_check is None:  # last_hidden_state == post_layernorm(last encoder output): LN applied once
            pre = out.hidden_states[-1]
            again = tower.vision_model.post_layernorm(pre)
            ln_check = {"max_abs_diff_vs_post_layernorm(encoder_out)": float((again - h).abs().max()),
                        "max_abs_diff_vs_encoder_out(no LN)": float((pre - h).abs().max())}
        feats[i:i + len(ims)] = h.float().cpu().numpy().astype(np.float16)
    feats.flush()
    stats = {"n_images": len(keys), "seconds": time.perf_counter() - t0, "batch": batch,
             "bytes": (cache_dir / "features.npy").stat().st_size, "dtype": "float16",
             "shape": [len(keys), C.N_PATCH, C.D_VIS], "layernorm_check": ln_check,
             "preprocessing": {k: getattr(proc, k, None) for k in ("size", "resample", "image_mean", "image_std",
                                                                   "do_resize", "do_rescale", "rescale_factor")},
             **peak_memory_mb()}
    del tower
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    stats["gpu_memory_after_release_mb"] = mb(torch.cuda.memory_allocated()) if dev.type == "cuda" else None
    return stats


class FeatureStore:
    """Read-only access to cached features by image key."""

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        self.arr = np.load(cache_dir / "features.npy", mmap_mode="r")
        keys = (cache_dir / "keys.txt").read_text().split()
        self.index = {k: i for i, k in enumerate(keys)}

    def get(self, keys, pool=False):
        x = torch.from_numpy(np.stack([self.arr[self.index[k]] for k in keys]).astype(np.float16))
        if pool:  # optional 49-token variant: average each 2x2 block of the 14x14 grid
            x = x.view(len(keys), 14, 14, -1).unflatten(1, (7, 2)).unflatten(3, (7, 2))
            x = x.float().mean(dim=(2, 4)).reshape(len(keys), 49, -1).half()
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/task2/data")
    ap.add_argument("--images", default="/tmp/pa3_images", help="COCO images root (split/COCO_split_id.jpg)")
    ap.add_argument("--cache", default="/tmp/pa3_feats")
    ap.add_argument("--vision", default=C.VISION_ID)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    keys = (Path(a.data) / "images.txt").read_text().split()
    cache = Path(a.cache)
    if (cache / "keys.txt").exists() and (cache / "keys.txt").read_text().split() == keys:
        print("features already cached")
        return
    dl = fetch_images(keys, a.images)
    if dl["n_errors"]:
        raise SystemExit(f"{dl['n_errors']} images failed to download, e.g. {dl['errors'][:3]}")
    with Timer() as t:
        st = cache_features(keys, a.images, cache, a.vision)
    (cache / "keys.txt").write_text("\n".join(keys) + "\n")
    write_json(a.stats or Path(a.data).parent / "feature_cache_stats.json", {"download": dl, "caching": st,
                                                                              "total_seconds": t.elapsed})
    print({"download_s": round(dl["seconds"], 1), **{k: st[k] for k in ("seconds", "bytes", "layernorm_check")}})


if __name__ == "__main__":
    main()
