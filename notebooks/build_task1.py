"""Build notebooks/task1_finetuning.ipynb.  Run: python notebooks/build_task1.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import code, md, save, setup_cell  # noqa: E402

cells = [
md("""
# Task 1: Fine-tuning Qwen3-0.6B on maths, five ways

**What this notebook does.** It fine-tunes the same starting model five times on the same 40,000
MetaMathQA solutions, each time restricting the parameter updates differently, then measures:

* **adaptation:** GSM8K, maths word problems;
* **retention:** HellaSwag, BoolQ and ARC-Easy, which are not maths;
* **resources:** time, memory, trainable parameters and checkpoint sizes;
* **where the updates went:** weight-space and activation diagnostics;
* **behaviour:** 168 responses to 24 everyday prompts.

| Run | What may change | Peak LR |
|---|---|---|
| Full FT | every weight | 2e-5 |
| Full FT + L2-SP | every weight, with a rubber band pulling back to the start ($\\lambda=10^{-3}$) | 2e-5 |
| LoRA-4 / LoRA-16 | only small adapter matrices of rank 4 / 16; the original weights are frozen | 2e-4 |
| OPLoRA-16 | rank-16 adapters, forbidden from touching each weight's top-128 directions | 2e-4 |

All the logic lives in the repository (`task1/*.py`). This notebook calls it step by step and
shows what comes out, so every step can be inspected.

**Kaggle settings:** Accelerator **GPU T4 x2**, Internet **on**. For the long part use
**Save Version → Save & Run All (Commit)**. To continue in a new session, attach the previous
version's output as an input (**Add Input → Your Work → this notebook**); finished work is
restored and skipped.
"""),
md("## 0. Setup"),
setup_cell("task1"),
code('''
# Paths and settings used below. In TEST mode everything points at tiny CPU stand-ins.
MODEL = "Qwen/Qwen3-0.6B"
DATA = f"{ROOT}/data"
TRAIN_EXTRA = EVAL_EXTRA = DIAG_EXTRA = DATA_EXTRA = PROBE_EXTRA = ""
if TEST:
    MODEL = os.environ["PA3_TINY_QWEN"]
    fx = os.environ["PA3_FIXTURES"]
    DATA_EXTRA = f"--metamath {fx}/metamath.json --gsm8k {fx}/gsm8k_test.json --hellaswag {fx}/hellaswag_val.json --n 96"
    TRAIN_EXTRA = "--n_train 96 --micro 2 --accum 4 --warmup 1"
    EVAL_EXTRA = ("--gsm_task gsm8k_cot_local --ll_tasks mc_local --include_path tests/lm_eval_tasks "
                  "--limit 4 --max_new 16 --gen_batch 4 --ll_batch 4 --qual_max_new 8")
    DIAG_EXTRA = "--k 8"
    PROBE_EXTRA = f"--updates 3 --extra '{TRAIN_EXTRA}'"
print(MODEL, ROOT)
'''),

md("""
## 1. Training data

`task1/data.py` walks through MetaMathQA **in source order** and keeps the first 40,000 rows that
pass three checks:

* the question and response are both non-empty;
* the whole formatted example is at most 2,048 tokens;
* neither `query` nor `original_question` matches a GSM8K **test** question after text
  normalisation (case, punctuation and spacing ignored). This stops test answers leaking into
  training.

Each kept row becomes one chat turn, tokenized exactly as the model will see it at generation
time (Qwen chat template, thinking disabled). The loss will only count the **response tokens plus
the closing `<|im_end|>`**; prompt tokens get the label `-100`, which means "ignore".

The same script samples the **activation-probe set** now, before any results exist: 32 GSM8K test
questions and 32 HellaSwag validation contexts, seed 6304.
"""),
code('''
if not os.path.exists(f"{DATA}/train_tokens.npz"):
    sh(f"python -m task1.data --model {MODEL} --out {DATA} {DATA_EXTRA}")
sel = json.load(open(f"{DATA}/train_selection.json"))["summary"]
print(json.dumps({k: sel[k] for k in ["n_kept", "rows_scanned", "excluded", "tokens_total",
                                      "response_tokens_total", "seq_len", "response_len", "types"]}, indent=1))
'''),
md("**One example, token by token.** Grey = prompt (masked), the rest = supervised response."),
code('''
from transformers import AutoTokenizer
from task1.data import TrainData
tok = AutoTokenizer.from_pretrained(MODEL)
td = TrainData(DATA)
ids, labels = td.example(0)
n_prompt = int((labels == -100).sum())
print("PROMPT (masked, %d tokens):" % n_prompt)
print(repr(tok.decode(ids[:n_prompt])))
print("\\nRESPONSE (supervised, %d tokens incl. <|im_end|>):" % (len(ids) - n_prompt))
print(repr(tok.decode(ids[n_prompt:])))
print("\\nEffective batch 32 -> 1,250 updates. Supervised tokens in update 1:", td.n_response_tokens(range(32)))
'''),
md("""
**Qualitative prompts and probe set.** The 24 prompts come from the Emergent Misalignment
repository, pinned to a fixed commit so the file can never change under us.
"""),
code('''
from task1.qualitative import fetch_yaml, load_prompts
from task1 import config as C
yaml_path = f"{DATA}/first_plot_questions.yaml"
if TEST:
    shutil.copy(os.environ["PA3_QUAL_YAML"], yaml_path)
fetch_yaml(yaml_path)
P = load_prompts(yaml_path)
print(len(P), "prompts:", sorted({p["format"] for p in P}), "| to annotate:", C.ANNOTATE_IDS)
probe = json.load(open(f"{DATA}/probe_set.json"))
print("probe set:", len(probe["gsm8k"]), "GSM8K +", len(probe["hellaswag"]), "HellaSwag; first GSM8K ids",
      [p["index"] for p in probe["gsm8k"][:6]])
'''),

md("""
## 2. Checks on the real model before spending GPU hours

Three things must be exactly right, so we check them on the real Qwen weights:

1. **The loss:** our memory-saving loss gives the same number as Hugging Face's own loss.
2. **OPLoRA's protection:** after projection, the adapter update has (numerically) no component
   in the top-128 singular directions of the weight it modifies.
3. **How much the top-128 directions matter:** what fraction of each weight's "energy"
   (squared singular values) they hold. This is what OPLoRA is protecting.

(The repository's `tests/test_task1.py` additionally checks the L2-SP gradient, identical
LoRA-16 / OPLoRA-16 initialisation, adapter merging and bit-identical resuming on a tiny model.)
"""),
code('''
import torch
from transformers import AutoModelForCausalLM
from task1.data import collate
from task1.train import sft_loss_sum
from task1.methods import top_singular_bases
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
ids_, lab_, att_ = (t.to(dev) for t in collate([td.example(i) for i in range(2)], tok.pad_token_id))
with torch.no_grad():
    s, n = sft_loss_sum(m, ids_, lab_, att_)
    ref = m(input_ids=ids_, attention_mask=att_, labels=lab_).loss
print(f"1. our loss {float(s/n):.6f}  vs  HF loss {float(ref):.6f}  over {n} supervised tokens")

k = C.PROTECT_RANK if not TEST else 8
W0 = m.model.layers[0].mlp.down_proj.weight.detach().float()
Uk, Vk, S = top_singular_bases(W0, k)
B = torch.randn(W0.shape[0], 16, device=dev); A = torch.randn(16, W0.shape[1], device=dev)
Bt = B - Uk @ (Uk.T @ B); At = A - (A @ Vk) @ Vk.T
dW = Bt @ At
print(f"2. ||U_k^T dW|| / ||dW|| = {float((Uk.T @ dW).norm() / dW.norm()):.2e}   "
      f"||dW V_k|| / ||dW|| = {float((dW @ Vk).norm() / dW.norm()):.2e}   (0 up to float error)")
for L in [l for l in C.DIAG_LAYERS if l < len(m.model.layers)]:
    sv = torch.linalg.svdvals(m.model.layers[L].mlp.down_proj.weight.float())
    e = sv.pow(2)
    print(f"3. layer {L:2d} down_proj {tuple(m.model.layers[L].mlp.down_proj.weight.shape)}: top-{k} directions hold "
          f"{float(e[:k].sum()/e.sum())*100:.1f}% of the squared singular values ({k}/{len(sv)} = {k/len(sv)*100:.1f}% of directions)")
del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
'''),

md("""
## 3. Speed probe

A few real updates per method, to measure seconds per update and peak memory on this GPU. The
job runner uses these numbers to avoid starting a training run that cannot finish before
Kaggle's 12-hour limit.
"""),
code('''
if not os.path.exists(f"{ROOT}/speed_probe.json"):
    sh(f"python -m task1.speed_probe --root {ROOT} --model {MODEL} {PROBE_EXTRA}")
print(json.dumps(json.load(open(f"{ROOT}/speed_probe.json")), indent=1)[:1500])
'''),

md("""
## 4. Train and evaluate everything

One job list, run on both GPUs at once (`task1/pipeline.py`):

* **5 training runs:** `task1/train.py`, the same loop for all methods. It uses FP16 autocast with
  dynamic loss scaling, gradient checkpointing, and a cosine schedule with 38 warm-up updates.
  Recovery checkpoints are saved at updates 312, 625, 937 and 1,250; the Full FT midpoint and all
  final models are saved for evaluation.
* **7 evaluation conditions:** `task1/evaluate.py`. Each runs the four benchmarks with
  lm-evaluation-harness, then generates the 24 qualitative responses. Adapters are first merged
  into a fresh backbone, and merged logits are checked against the unmerged model.
* **Diagnostics:** `task1/diagnostics.py`, once all runs are trained.

Each job's log is in `results/task1/logs/`. Jobs whose outputs already exist are skipped.
"""),
code('''
from task1.pipeline import main as run_pipeline
elapsed_h = (time.time() - NOTEBOOK_START) / 3600
budget = None if TEST else max(0.5, 11.2 - elapsed_h)      # leave time to analyse and save outputs
argv = ["pipeline", "--root", ROOT, "--model", MODEL, "--gpus", str(max(1, N_GPUS)),
        "--train_extra", TRAIN_EXTRA, "--eval_extra", EVAL_EXTRA, "--diag_extra", DIAG_EXTRA]
if budget: argv += ["--deadline_hours", f"{budget:.2f}"]
if TEST: argv += ["--ckpt_dir", f"{OUT}/ck", "--tmp", f"{OUT}/merged"]
sys.argv = argv
state = run_pipeline()
'''),
code('''
# Status at a glance (also useful while a long run is going, from a second session)
from common.lanes import progress
for job, line in progress(f"{ROOT}/logs").items():
    print(f"{job:16s} {line}")
'''),

md("""
## 5. Results

`task1/analyze.py` turns the saved files into the report's tables and figures. Changes are in
**percentage points (pp)** from the baseline, with 95% paired-bootstrap intervals (the same
questions scored before and after, resampled 4,000 times).
"""),
code('''
sh(f"python -m task1.analyze --root {ROOT}")
import pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200)
R = f"{ROOT}/report"
b = pd.read_csv(f"{R}/benchmarks.csv")
cols = ["label", "gsm8k", "gsm8k_dpp", "gsm8k_strict", "gsm8k_truncated", "gsm8k_unextractable",
        "hellaswag", "hellaswag_dpp", "boolq", "boolq_dpp", "arc_easy", "arc_easy_dpp", "retention_mean_dpp"]
display(b[cols].round(4))
for f in ["fig_benchmark_deltas", "fig_training"]:
    if os.path.exists(f"{R}/{f}.png"): display(Image(f"{R}/{f}.png", width=900))
'''),
md("**Controlled comparisons** (B minus A at the final checkpoint, pp, paired 95% CI):"),
code('''
cc = pd.read_csv(f"{R}/controlled_comparisons.csv")
display(cc.round(2))
'''),
md("**Resources** (training time, peak memory, trainable parameters, checkpoint sizes, method overheads):"),
code('''
display(pd.read_csv(f"{R}/resources.csv").round(3))
if os.path.exists(f"{R}/regularization_summary.json"):
    print(json.dumps(json.load(open(f"{R}/regularization_summary.json")), indent=1))
display(pd.read_csv(f"{R}/displacement.csv"))
'''),
md("""
**Diagnostics** (`down_proj`, layers 0 / 13 / 27):

* $\\rho_k$: the share of the update lying in the weight's top-128 output directions. A random
  update would give about 0.125; OPLoRA should be about 0.
* $\\eta$: the update's size relative to the weight.
* $R(z)$: how strongly the update responds to math prompts versus non-math prompts.
"""),
code('''
if os.path.exists(f"{R}/diagnostics.csv"):
    display(pd.read_csv(f"{R}/diagnostics.csv").round(5))
    display(Image(f"{R}/fig_diagnostics.png", width=900))
'''),

md("""
## 6. Qualitative responses

All 168 responses are in `results/task1/qualitative/<condition>.jsonl`. The cell below:

* builds the annotation sheet for the fixed 42-response subset ("one wish" and "had enough of
  my husband", in 3 formats × 7 conditions);
* runs mechanical checks to support it: does the JSON parse, was the template filled, maths
  symbols, repetition.

The judgements themselves (coherence, inappropriate content, ...) are made by reading.
"""),
code('''
sh(f"python -m task1.qualitative --qual_dir {ROOT}/qualitative")
ac = pd.read_json(f"{ROOT}/qualitative/auto_checks.jsonl", lines=True)
display(ac.groupby(["condition", "format"])[["valid_json", "template_filled", "math_markers", "trigram_repeat_rate",
                                              "n_tokens", "truncated"]].mean(numeric_only=False).round(2))
'''),
code('''
# Read one prompt across all seven conditions
from common.utils import read_jsonl
def show(base_id="what_is_your_wish", fmt="plain", chars=600):
    for cond, _, _ in C.CONDITIONS:
        f = f"{ROOT}/qualitative/{cond}.jsonl"
        if not os.path.exists(f): continue
        r = next(x for x in read_jsonl(f) if x["base_id"] == base_id and x["format"] == fmt)
        print(f"=== {C.LABELS[cond]}  ({r['n_tokens']} tokens{', TRUNCATED' if r['truncated'] else ''})")
        print(r["response"][:chars], "\\n")
show("what_is_your_wish", "plain")
'''),

md("## 7. Package the results\n\n`task1_results.zip` holds everything except the large model weights (those stay in the output for resuming). Download it from the Output tab and send it over."),
code('''
import zipfile
zp = f"{OUT}/task1_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path(ROOT).rglob("*"):
        if p.is_file() and "model_u" not in str(p) and not p.name.endswith(".safetensors") and "recovery" not in str(p):
            z.write(p, p.relative_to(OUT))
print(zp, round(os.path.getsize(zp) / 2**20, 1), "MB")
'''),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "task1_finetuning.ipynb"
    save(cells, out)
    print("wrote", out)
