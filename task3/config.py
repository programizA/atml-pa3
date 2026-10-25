"""Every Task 3 setting from the assignment, in one place."""
from __future__ import annotations

SEED = 6304
DATASETS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
N_DEV = 200
N_LIVE = 50
BUDGETS = (1, 2, 4, 8, 16, 32)
N_ORDERS = 10
N_BOOT = 1000
NO_ANSWER = "NO_ANSWER"

# Adaptive sampling
CHECK_AT = (4, 8, 16)
TAUS = (0.5, 0.625, 0.75, 0.875, 1.0)
DELTAS = (0.0, 0.1, 0.25, 0.5)
DEV_TOLERANCE_PP = 1.0

# Live generation (Step 4)
LIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LIVE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
LIVE_LIMITS = {"single": 3072, "parallel": 1024, "initial": 1024, "revision": 2048}
N_PARALLEL = 3
