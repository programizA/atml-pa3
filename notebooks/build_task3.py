"""Build notebooks/task3_test_time_compute.ipynb.  Run: python notebooks/build_task3.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import code, md, save, setup_cell  # noqa: E402

cells = [
md("""
# Task 3: Test-time compute: sampling, selection and adaptive inference

**The question.** Given 32 sampled solutions per question, how much does spending more compute at
answer time help, and how should that compute be spent?

* **Steps 1-2 (offline, CPU):** with N = 1, 2, …, 32 candidates, how often is *some* candidate
  right (oracle coverage), and how often does a selection rule *pick* a right one? The rules are
  majority vote, model confidence, visual attention, and combinations.
* **Step 3 (offline):** what do the confidence and vision signals actually track?
* **Step 4 (GPU, live):** on 50 MATH questions with the same 3,072-token output allowance,
  compare one long answer, three short independent answers with a vote, and answer-then-revise.
* **Step 5 (offline):** stop sampling early once the answers agree, and measure what that saves
  and what it costs.

**Inputs.** Upload the supplied `ATML-PA3-Data` folder as a private Kaggle dataset named
`atml-pa3-data` and attach it (**Add Input**). Kaggle settings: GPU T4 x2, Internet on. Only
section 4 uses the GPU.
"""),
md("## 0. Setup"),
setup_cell("task3"),
code('''
found = sorted(glob.glob("/kaggle/input/**/qwen25-math", recursive=True))
RAW = os.path.dirname(found[0]) if found else None      # the folder holding qwen25-math/ and qwen25vl-mathvista/
META_URL = ("https://raw.githubusercontent.com/lupantech/MathVista/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb/"
            "data/testmini.json")
META = f"{ROOT}/mathvista_testmini.json"
LIVE_MODEL, LIVE_REV, LIVE_EXTRA = "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", ""
if TEST:
    RAW = os.environ["PA3_T3_RAW"]; shutil.copy(os.environ["PA3_MV_META"], META)
    LIVE_MODEL, LIVE_REV, LIVE_EXTRA = os.environ["PA3_TINY_QWEN"], "", "--max_new_override 12"
elif not os.path.exists(META):
    import urllib.request; urllib.request.urlretrieve(META_URL, META)
print("rollout data:", RAW)
'''),

md("""
## 1. Load, join and verify; answers and correctness (`task3/data.py`)

* **Join:** the three tables are joined on `(dataset, sample_id, rollout_id)`.
* **Verify:** every per-token list has exactly one entry per retained generated token, and each
  question has 32 rollouts.
* **Group answers** without looking at the gold answer:
  * MATH: math-verify decides mathematical equivalence, e.g. `\\frac{1}{2}` = `0.5`.
  * MathVista: the official rules give option letters, integers, floats rounded to the
    question's precision, or lists.
  * Missing extractions form one `NO_ANSWER` group. It still gets votes and costs tokens, and it
    is always wrong.
* **Splits:** sort the IDs, permute them with seed 6304, then take 200 dev / 800 held-out
  questions. Ten seeded orders of the 32 rollouts are fixed per question.
"""),
code('''
if not os.path.exists(f"{ROOT}/mathvista/rollouts.parquet"):
    sh(f"python -m task3.data --raw {RAW} --meta {META} --out {ROOT}")
for d in ["math", "mathvista"]:
    r = json.load(open(f"{ROOT}/{d}/data_report.json"))
    print(d, {k: r[k] for k in ["rows", "questions", "mismatches", "capped", "null_extractions", "retained_tokens",
                                 "rollout_accuracy", "scorer"]})
'''),

md("""
## 2. Best-of-N simulation (`task3/selection.py`)

For each question, order and budget N, every rule sees only the first N rollouts of that order.

* **Self-certainty** $K_i$: the mean over tokens of KL(uniform ‖ model distribution). High means
  the model's predictions were sharp. It is a confidence signal, not a probability of being right.
* **Visual engagement** $V_i$: the share of attention on image tokens.
* **VisConf** $Z_i = V_i \\cdot C_i$, where $C_i$ is self-certainty rescaled to [0, 1] using the
  5th-95th percentiles of the *current candidates'* tokens only.
* **RWC:** each candidate votes for its answer with weight $(N - \\text{rank})/(N-1)$, so better
  scored rollouts count more.
"""),
code('''
if not os.path.exists(f"{ROOT}/mathvista/selection_results.parquet"):
    sh("python -m task3.selection --root " + ROOT)
sh("python -m task3.signals --root " + ROOT + " > /dev/null")
sh("python -m task3.adaptive --root " + ROOT)
'''),

md("""
## 3. Results of the offline experiments

The figures and tables are on the 800 held-out questions. Differences come with 1,000 paired
question-level bootstrap intervals.
"""),
code('''
sh("python -m task3.analyze --root " + ROOT + " > /dev/null")
import pandas as pd
from IPython.display import Image, display
R = f"{ROOT}/report"
m = pd.read_csv(f"{R}/methods_by_budget.csv")
display(m[m.N.isin([8, 32])].pivot_table(index="method", columns=["dataset", "N"], values="acc").round(3))
display(pd.read_csv(f"{R}/decomposition.csv").query("N in [4, 8, 32]").round(3))
display(pd.read_csv(f"{R}/adaptive.csv").round(3))
for f in ["fig_scaling", "fig_adaptive", "fig_trajectories"]:
    display(Image(f"{R}/{f}.png", width=900))
'''),

md("""
## 4. Live generation: one long answer vs three short ones vs answer-then-revise (GPU)

`task3/live.py` covers the first 50 held-out MATH questions. It loads Qwen2.5-3B-Instruct at the
README's pinned revision in FP16, with batch size 1 and the export's sampling settings, and gives
every call its own seed. On two GPUs the questions are split between them; all three strategies
for a question run on the same GPU, so their costs compare like for like. Each call records its
prompt, response, input/output tokens, truncation flag, seed and wall time (model loading
excluded).
"""),
code('''
from common.lanes import Job, run_jobs
from common.utils import read_jsonl
n = max(1, N_GPUS)
def done(i):
    f = f"{ROOT}/live/calls_shard{i}.jsonl"
    ids = json.load(open(f"{ROOT}/live_questions.json"))["ids"]
    if TEST: ids = ids[:3]
    want = {(q, s) for qi, q in enumerate(ids) if qi % n == i for s in range(5)}
    return os.path.exists(f) and want <= {(r["sample_id"], r["slot"]) for r in read_jsonl(f)}
if TEST:   # three questions are enough to test the plumbing
    lq = json.load(open(f"{ROOT}/live_questions.json")); lq["ids"] = lq["ids"][:3]
    json.dump(lq, open(f"{ROOT}/live_questions.json", "w"))
rev = f"--revision {LIVE_REV}" if LIVE_REV else "--revision ''"
jobs = [Job(f"live_{i}", f"python -m task3.live --root {ROOT} --model {LIVE_MODEL} {rev} --shard {i}/{n} {LIVE_EXTRA}",
            done=lambda i=i: done(i), gpu_pref=i) for i in range(n)]
print(run_jobs(jobs, n, f"{ROOT}/logs", cwd=REPO))
sh(f"python -m task3.live --root {ROOT} --summarize")
sh("python -m task3.analyze --root " + ROOT + " > /dev/null")
display(pd.read_csv(f"{R}/live.csv").round(3))
print(json.dumps(json.load(open(f"{ROOT}/live/live_summary.json"))["refinement"], indent=1))
'''),

md("## 5. Package the results\n\n`task3_results.zip` holds everything except the large token arrays. Download it from the Output tab and send it over."),
code('''
import zipfile
zp = f"{OUT}/task3_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path(ROOT).rglob("*"):
        if p.is_file() and p.name not in ("tokens.npz", "texts.parquet"):
            z.write(p, p.relative_to(OUT))
print(zp, round(os.path.getsize(zp) / 2**20, 1), "MB")
'''),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "task3_test_time_compute.ipynb"
    save(cells, out)
    print("wrote", out)
