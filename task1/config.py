"""Every Task 1 setting from the assignment, in one place. Scripts import from here so the five
runs cannot drift apart."""
from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 6304

# Data
N_TRAIN = 40_000
MAX_LEN = 2_048
METAMATH_ID = "meta-math/MetaMathQA"
GSM8K_ID = ("openai/gsm8k", "main")

# Optimisation (shared by all five runs)
MICRO_BATCH = 2
GRAD_ACCUM = 16
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM          # 32
TOTAL_UPDATES = N_TRAIN // EFFECTIVE_BATCH           # 1250
WARMUP_UPDATES = 38
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
RECOVERY_UPDATES = (312, 625, 937, 1250)             # 9,984 / 20,000 / 29,984 / 40,000 examples
MIDPOINT_UPDATE = 625
FINAL_UPDATE = 1250

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_DROPOUT = 0.05
PROTECT_RANK = 128

# The five runs. "lam" is the L2-SP strength (0 = no penalty).
METHODS = {
    "full":   dict(kind="full", lr=2e-5, lam=0.0),
    "l2sp":   dict(kind="full", lr=2e-5, lam=1e-3),
    "lora4":  dict(kind="lora", lr=2e-4, lam=0.0, r=4,  alpha=8),
    "lora16": dict(kind="lora", lr=2e-4, lam=0.0, r=16, alpha=32),
    "oplora": dict(kind="oplora", lr=2e-4, lam=0.0, r=16, alpha=32, k=PROTECT_RANK),
}

# Seven evaluation conditions: (name, run, update)
CONDITIONS = [
    ("baseline", None, 0),
    ("full_mid", "full", MIDPOINT_UPDATE),
    ("full", "full", FINAL_UPDATE),
    ("l2sp", "l2sp", FINAL_UPDATE),
    ("lora4", "lora4", FINAL_UPDATE),
    ("lora16", "lora16", FINAL_UPDATE),
    ("oplora", "oplora", FINAL_UPDATE),
]
LABELS = {"baseline": "Baseline", "full_mid": "Full FT (mid)", "full": "Full FT",
          "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora": "OPLoRA-16"}

# Diagnostics
DIAG_LAYERS = (0, 13, 27)
N_PROBE = 32

# Qualitative prompts: pinned commit of the emergent-misalignment repository
QUAL_YAML_URL = ("https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
                 "80c11967c07a328e7d7d43d13ce6847ae44dbcc9/evaluation/first_plot_questions.yaml")
QUAL_MAX_NEW = 512
ANNOTATE_IDS = ("what_is_your_wish", "enough_of_my_husband")   # first and last plain questions

# Benchmarks
GSM8K_MAX_NEW = 1024
BENCH = {  # task -> (primary metric, raw metric)
    "gsm8k_cot": ("exact_match,flexible-extract", "exact_match,strict-match"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "boolq": ("acc,none", "acc,none"),
    "arc_easy": ("acc_norm,none", "acc,none"),
}
