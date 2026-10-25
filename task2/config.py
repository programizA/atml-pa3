"""Every Task 2 setting from the assignment, in one place."""
from __future__ import annotations

SEED = 6304
LM_ID = "Qwen/Qwen3-0.6B"
VISION_ID = "google/siglip-base-patch16-224"
N_PATCH, D_VIS, D_LM = 196, 768, 1024

# Data sizes
N_CAP_TRAIN_IMAGES, CAPS_PER_TRAIN_IMAGE, N_CAP_DEV = 6000, 3, 96
VQA_COUNTS = {  # pairs per answer type
    "train": {"yes/no": 640, "number": 256, "other": 640},
    "dev": {"yes/no": 12, "number": 8, "other": 12},
    "test": {"yes/no": 36, "number": 24, "other": 36},
}
GROUND_EVAL = {"dev": {"presence": 12, "counting": 12}, "test": {"presence": 32, "counting": 32}}
MIN_AGREE = 8
POOL_PAIRS_PER_CATEGORY = 300          # reusable training pools (pairs may be re-drawn)
PRESENCE_MIN_AREA = 0.01               # a "present" object covers >= 1% of the image
COUNT_MIN_AREA = 0.005                 # every counted instance >= 0.5% (else the count is ambiguous)
COUNT_RANGE = (1, 4)

# Prompts (content between the visual tokens and the end of the user turn)
CAPTION_PROMPT = "Write a short caption for this image."
SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or number."

# Token caps (content tokens)
Q_CAP, CAPTION_CAP, ANSWER_CAP = 80, 64, 16

# Optimisation
MICRO_BATCH, GRAD_ACCUM = 8, 2
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM          # 16
WEIGHT_DECAY, CLIP_NORM = 0.01, 1.0
WARMUP_FRAC, MIN_LR_FRAC = 0.05, 0.10
STAGES = [  # name, updates, connector LR, LoRA LR (None = LoRA off), mixture
    dict(name="caption", updates=2250, lr_conn=1e-3, lr_lora=None, mix=None),
    dict(name="instruct", updates=1500, lr_conn=1e-4, lr_lora=2e-5,
         mix={"presence": 0.50, "counting": 0.25, "caption": 0.25}),
    dict(name="vqa", updates=750, lr_conn=5e-5, lr_lora=1e-5,
         mix={"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}),
]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
CKPT_EVERY = 250
DEV_EVERY = 500
STAGE_ENDS = (2250, 3750, 4500)

# Final evaluation
GEN_MAX_NEW, GEN_BATCH = 16, 8
N_BOOT = 4000
CONDITIONS = ("correct", "complementary", "mismatched", "no_image")

# URLs (official hosts)
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMG_URL = "http://images.cocodataset.org/{split}/COCO_{split}_{id:012d}.jpg"
VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_ZIPS = {
    "q_train": ("v2_Questions_Train_mscoco.zip", "v2_OpenEnded_mscoco_train2014_questions.json"),
    "q_val": ("v2_Questions_Val_mscoco.zip", "v2_OpenEnded_mscoco_val2014_questions.json"),
    "a_train": ("v2_Annotations_Train_mscoco.zip", "v2_mscoco_train2014_annotations.json"),
    "a_val": ("v2_Annotations_Val_mscoco.zip", "v2_mscoco_val2014_annotations.json"),
    "p_train": ("v2_Complementary_Pairs_Train_mscoco.zip", "v2_mscoco_train2014_complementary_pairs.json"),
    "p_val": ("v2_Complementary_Pairs_Val_mscoco.zip", "v2_mscoco_val2014_complementary_pairs.json"),
}

PLURALS = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "pairs of skis",
           "scissors": "pairs of scissors", "bus": "buses", "wine glass": "wine glasses", "knife": "knives",
           "sandwich": "sandwiches", "couch": "couches", "bench": "benches", "toothbrush": "toothbrushes",
           "skateboard": "skateboards", "glass": "glasses", "broccoli": "pieces of broccoli",
           "hair drier": "hair driers", "potted plant": "potted plants", "sports ball": "balls"}

# ---- CPU smoke-test scale (PA3_T2_SMALL=1): same code paths, tiny training budget ----
import os as _os

if _os.environ.get("PA3_T2_SMALL") == "1":
    N_CAP_TRAIN_IMAGES = 40
    VQA_COUNTS = {"train": {"yes/no": 8, "number": 4, "other": 8},
                  "dev": {"yes/no": 3, "number": 2, "other": 3},
                  "test": {"yes/no": 6, "number": 4, "other": 6}}
    GROUND_EVAL = {"dev": {"presence": 3, "counting": 3}, "test": {"presence": 6, "counting": 6}}
    N_CAP_DEV = 12
    MICRO_BATCH, GRAD_ACCUM = 2, 2
    EFFECTIVE_BATCH = 4
    STAGES[0]["updates"], STAGES[1]["updates"], STAGES[2]["updates"] = 6, 4, 4
    CKPT_EVERY, DEV_EVERY = 3, 5
    STAGE_ENDS = (6, 10, 14)
    POOL_PAIRS_PER_CATEGORY = 10
