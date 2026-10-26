"""Figures that need the COCO images (run where the images can be downloaded, e.g. Kaggle CPU).

    python -m task2.figures --root results/task2 --images /tmp/pa3_images

1. Label audit sheets: every annotation-derived test pair (32 presence + 32 counting) with its
   generated question and answers, numbered, and *without* model predictions, so the labels can be
   judged independently. Also writes label_audit.csv: one row per test image with an empty
   "doubtful" column to fill in (y = the generated answer is wrong or ambiguous for this image).
2. The aligned complementary-pair examples figure (rule in task2.analyze.select_examples), for the
   main and optional runs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import plotstyle as ps
from common.utils import read_json, write_json
from task2.analyze import fig_examples, select_examples
from task2.features import fetch_images, image_path


def audit_sheets(root, images, per_page=16):
    import matplotlib.pyplot as plt
    from PIL import Image

    sel = read_json(Path(root) / "data" / "selection.json")
    pairs = sel["ground_test"]
    out = Path(root) / "data"
    fetch_images(sorted({p[k] for p in pairs for k in ("image_a", "image_b")}), images)
    with open(out / "label_audit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "number", "kind", "side", "image", "question", "generated_answer", "doubtful(y/n)", "note"])
        for i, p in enumerate(pairs):
            for s in "ab":
                w.writerow([p["pair_id"], i, p["kind"], s, p[f"image_{s}"], p["question"], p[f"answer_{s}"], "", ""])
    ps.setup()
    for kind in ("presence", "counting"):
        sub = [(i, p) for i, p in enumerate(pairs) if p["kind"] == kind]
        for page in range(0, len(sub), per_page):
            chunk = sub[page:page + per_page]
            fig, axes = plt.subplots(4, 8, figsize=(16, 9.2))
            for ax in axes.flat:
                ax.axis("off")
            for j, (i, p) in enumerate(chunk):
                for k, s in enumerate("ab"):
                    ax = axes[j // 4, (j % 4) * 2 + k]
                    ax.imshow(Image.open(image_path(images, p[f"image_{s}"])).convert("RGB"))
                    ax.set_title(f"#{i}{s}: {p['category']}\n-> {p[f'answer_{s}']}", fontsize=8, loc="center",
                                 fontweight="normal")
            fig.suptitle(f"Annotation-derived {kind} test pairs (generated labels, no model predictions)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"label_audit_{kind}_{page // per_page + 1}.png", dpi=110)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task2")
    ap.add_argument("--images", default="/tmp/pa3_images")
    a = ap.parse_args()
    audit_sheets(a.root, a.images)
    sel = read_json(Path(a.root) / "data" / "selection.json")
    for run in ("run", "run_pool49"):
        r = Path(a.root) / run
        if not (r / "test_predictions.jsonl").exists():
            continue
        picks = select_examples(r)
        write_json(r / "report" / "example_picks.json", picks)
        allp = {p["pair_id"]: p for n in ("ground_test", "vqa_test") for p in sel[n]}
        fetch_images(sorted({allp[x["pair_id"]][k] for x in picks for k in ("image_a", "image_b")}), a.images)
        fig_examples(r, r / "report", a.images, picks)
    print("done")


if __name__ == "__main__":
    main()
