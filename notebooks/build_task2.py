"""Build notebooks/task2_vlm.ipynb.  Run: python notebooks/build_task2.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import code, md, save, setup_cell  # noqa: E402

cells = [
md("""
# Task 2: A vision-language model from SigLIP + Qwen3-0.6B, and does it really look?

**The model.** A frozen SigLIP image encoder turns each image into 196 patch features (a 14×14
grid, 768 numbers each). A small trainable **connector** (768 → 1024 → GELU → 1024) maps them
into Qwen's word-embedding space. They are inserted into the chat as 196 "visual tokens" before
the question. Qwen's own weights never change; from stage 2 on, small LoRA adapters on its
attention layers are trained.

**The curriculum.** One continuous run of 4,500 updates, 16 examples each:

| Stage | Updates | Trains | Data per draw |
|---|---|---|---|
| 1. Caption alignment | 2,250 | connector | captions (2 passes over 18,000) |
| 2. Visual instruction tuning | 1,500 | connector + LoRA | presence 50%, counting 25%, caption replay 25% |
| 3. Natural VQA | 750 | connector + LoRA | VQAv2 70%, presence / counting / captions 10% each |

**The grounding test.** Every question comes as a *complementary pair*: the same question on two
images with different answers, so the wording alone cannot get both right. Each test question is
answered with its own image, with the partner's image, with an unrelated image, and with no image.

**Kaggle settings:** GPU T4 x2, Internet on. The main run uses GPU 0. With `RUN_OPTIONAL = True`
the optional 49-token run trains on GPU 1 at the same time.
"""),
md("## 0. Setup"),
setup_cell("task2"),
code('''
RUN_OPTIONAL = True          # 49-visual-token variant on the second GPU (optional part of the task)
LM = "Qwen/Qwen3-0.6B"; VISION_ARG = ""
RAW = "/tmp/pa3_raw"; IMAGES = "/tmp/pa3_images"; CACHE = f"{ROOT}/cache"
if TEST:
    LM = os.environ["PA3_TINY_QWEN"]; VISION_ARG = f"--vision {os.environ['PA3_TINY_SIGLIP']}"
    RAW = os.environ["PA3_FX2"]; IMAGES = f"{RAW}/images"
DATA = f"{ROOT}/data"
'''),

md("""
## 1. Data selection

`task2/data.py` downloads the COCO-2014 caption and instance annotations and the VQAv2
questions, answers and complementary-pair lists from their official hosts (about 300 MB). It then
selects every split with seed 6304:

* **Captions:** 6,000 train images × their first 3 captions = 18,000 training examples, and 96
  val images × 1 caption for development.
* **Presence pairs**, e.g. "Does this image contain a dog?": one image where a dog is annotated
  and covers ≥1% of the picture, one with no dog annotation at all.
* **Counting pairs**, e.g. "How many dogs can you see?": two images with different dog counts
  (1-4). An image only qualifies if no dog is a "crowd" region and every dog covers ≥0.5% of the
  picture; otherwise the true count is ambiguous.
* **VQAv2 pairs:** the same question on two images whose consensus answers differ, with ≥8 of 10
  annotators agreeing on each. Counts per answer type follow the assignment.

All development and test images come from val2014 and are used once, so the evaluation sets are
disjoint from training and from each other. The isolation report below checks that.
"""),
code('''
if not os.path.exists(f"{DATA}/selection.json"):
    if not TEST:
        sh(f"python -m task2.data download --raw {RAW}")
    sh(f"python -m task2.data select --raw {RAW} --out {DATA}")
s = json.load(open(f"{DATA}/selection_summary.json"))
print(json.dumps({k: s[k] for k in ["sizes", "n_images", "n_train_images", "n_eval_images", "pool_categories", "notes"]}, indent=1))
print("isolation ok:", s["isolation"]["ok"], "| train/eval overlaps:", s["isolation"]["train_eval_overlap"])
'''),

md("""
## 2. Inspect the generated labels (before any model exists)

COCO annotations sometimes miss objects or contain ambiguous counts. The grid below shows the
first 8 presence and 8 counting test pairs with their generated answers, so the labels can be
checked by eye. It downloads just those images.
"""),
code('''
from task2.features import fetch_images, image_path
from common.utils import read_json
import matplotlib.pyplot as plt
from PIL import Image
sel = read_json(f"{DATA}/selection.json")
show = [p for p in sel["ground_test"] if p["kind"] == "presence"][:8] + [p for p in sel["ground_test"] if p["kind"] == "counting"][:8]
fetch_images(sorted({p[k] for p in show for k in ("image_a", "image_b")}), IMAGES)
fig, axes = plt.subplots(4, 8, figsize=(16, 8.5))
for i, p in enumerate(show):
    for j, side in enumerate("ab"):
        ax = axes[(i // 4) , (i % 4) * 2 + j]
        ax.imshow(Image.open(image_path(IMAGES, p[f"image_{side}"])).convert("RGB")); ax.axis("off")
        ax.set_title(f"{p['question'][:34]}\\n-> {p[f'answer_{side}']}", fontsize=7)
plt.tight_layout(); plt.savefig(f"{DATA}/label_inspection.png", dpi=110); plt.show()
'''),

md("""
## 3. What one training example looks like

Below is the exact token sequence for one VQAv2 example. Grey runs of `<|image_pad|>` are the
196 slots the connector's outputs are written into. Only the answer and the closing `<|im_end|>`
are supervised. The **no image** version drops the slots (and their newline) and rebuilds
positions; the text is identical.
"""),
code('''
from transformers import AutoTokenizer
from task2.model import Encoder
from task2.common import pair_examples
tok = AutoTokenizer.from_pretrained(LM)
enc = Encoder(tok)
ex = pair_examples(sel["vqa_train"][0])[0]
e = enc.encode(ex)
txt = tok.decode(e["ids"]).replace("<|image_pad|>" * 196, "<|image_pad|> x 196")
print(txt)
print("supervised tokens:", repr(tok.decode([t for t, l in zip(e["ids"], e["labels"]) if l != -100])))
print("sequence length with image:", len(e["ids"]), "| without image:", len(enc.encode(ex, image=False)["ids"]))
'''),

md("""
## 4. Feature caching, training and evaluation

`task2/pipeline.py` runs:

1. **features:** downloads the selected images (about 9,600) from the COCO host, runs the frozen
   SigLIP tower once per image in FP16, and stores the 196×768 features in one FP16 file. It then
   frees the encoder. Its time and memory are logged separately from training.
2. **train:** the three-stage curriculum. Before update 1 it calibrates the connector's output
   scale on training data only: median text-embedding norm divided by the median norm of the
   fresh connector's outputs. It then evaluates on the dev suites at update 0, every 500 updates
   and at each stage end, and saves resumable checkpoints every 250 updates.
3. **evaluate:** the final test under the four image conditions.
"""),
code('''
from task2.pipeline import main as run_pipeline
budget = None if TEST else max(0.5, 11.4 - (time.time() - NOTEBOOK_START) / 3600)
argv = ["pipeline", "--root", ROOT, "--lm", LM, "--images", IMAGES, "--cache", CACHE, "--gpus", str(max(1, N_GPUS))]
if VISION_ARG: argv += ["--vision_arg", VISION_ARG]
if RUN_OPTIONAL and (N_GPUS >= 2 or TEST): argv += ["--optional"]
if budget: argv += ["--deadline_hours", f"{budget:.2f}"]
sys.argv = argv
state = run_pipeline()
print(json.dumps(json.load(open(f"{ROOT}/run/calibration.json")), indent=1))
'''),

md("## 5. Results\n\nLearning curves (stage boundaries dotted), final tables with pair-bootstrap 95% intervals, and aligned complementary-pair examples chosen by a fixed rule (see `task2/analyze.py`)."),
code('''
import pandas as pd
from IPython.display import Image as Img, display
for run in ["run", "run_pool49"]:
    if not os.path.exists(f"{ROOT}/{run}/test_summary.json"): continue
    print("=" * 30, run)
    sh(f"python -m task2.analyze --root {ROOT} --run {ROOT}/{run} --images {IMAGES}")
    R = f"{ROOT}/{run}/report"
    display(pd.read_csv(f"{R}/final_results.csv").round(3))
    display(pd.read_csv(f"{R}/advantages.csv").round(3))
    for f in ["fig_curves", "fig_by_type", "fig_examples"]:
        if os.path.exists(f"{R}/{f}.png"): display(Img(f"{R}/{f}.png", width=950))
'''),

md("## 6. Package the results\n\n`task2_results.zip` excludes the feature cache and recovery checkpoints. Download it from the Output tab and send it over."),
code('''
import zipfile
zp = f"{OUT}/task2_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path(ROOT).rglob("*"):
        if p.is_file() and "/cache/" not in str(p) and "recovery" not in str(p):
            z.write(p, p.relative_to(OUT))
print(zp, round(os.path.getsize(zp) / 2**20, 1), "MB")
'''),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "task2_vlm.ipynb"
    save(cells, out)
    print("wrote", out)
