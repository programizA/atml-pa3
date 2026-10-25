# ATML PA3: Fine-tuning, Vision-Language Grounding, and Test-Time Compute

Code for EE-5102/CS-6304 Programming Assignment 3 (Fall 2026). Each task is a self-contained Python
package plus one guided Kaggle notebook that calls it step by step.

| Task | Package | Notebook | GPU |
|---|---|---|---|
| 1. LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA-16) | `task1/` | `notebooks/task1_finetuning.ipynb` | ~12-16 h on T4 x2 |
| 2. SigLIP-Qwen VLM and visual grounding | `task2/` | `notebooks/task2_vlm.ipynb` | ~3-4 h on T4 |
| 3. Test-time compute | `task3/` | `notebooks/task3_test_time_compute.ipynb` | offline on CPU; ~2 h live generation |

## Setup

```bash
pip install -r requirements.txt     # PyTorch itself is assumed preinstalled (Kaggle GPU image)
```

Every script uses seed 6304. All settings from the assignment live in `task*/config.py`.

## Reproducing each task

All commands run from the repository root and write to `results/<task>/`.

### Task 1
```bash
python -m task1.data                       # first 40,000 eligible MetaMathQA rows, probe set
python -m task1.speed_probe                # optional: seconds/update and peak memory per method
python -m task1.pipeline --gpus 2          # 5 trainings, 7 evaluation conditions, diagnostics
python -m task1.qualitative                # 42-response annotation sheet + automatic checks
python -m task1.analyze                    # tables and figures in results/task1/report/
```
Individual steps: `python -m task1.train --method {full,l2sp,lora4,lora16,oplora}`,
`python -m task1.evaluate --condition {baseline,full_mid,full,l2sp,lora4,lora16,oplora}`,
`python -m task1.diagnostics`.

### Task 2
```bash
python -m task2.data download && python -m task2.data select     # COCO 2014 + VQAv2 splits (seed 6304)
python -m task2.pipeline --gpus 2 --optional                      # features, 3-stage training, final eval (+49 tokens)
python -m task2.analyze
```

### Task 3
```bash
python -m task3.data --raw <ATML-PA3-Data> --meta <MathVista data/testmini.json>
python -m task3.selection && python -m task3.signals && python -m task3.adaptive
python -m task3.live --shard 0/1 && python -m task3.live --summarize        # GPU
python -m task3.analyze
```

## Tests

CPU checks on tiny stand-in models (same architecture, random weights):
```bash
python -m tests.tiny && python -m tests.fixtures && python -m task1.data --model /tmp/tiny/qwen --out /tmp/t1/data \
    --metamath /tmp/fx/metamath.json --gsm8k /tmp/fx/gsm8k_test.json --hellaswag /tmp/fx/hellaswag_val.json --n 96
python -m tests.test_task1                 # loss, L2-SP gradient, OPLoRA projection/init, merge, bit-exact resume
PA3_T2_SMALL=1 python -m tests.test_task2  # visual interface, frozen LM, connector gradients, resume
```

## Repository layout

```
common/      seeding, I/O, timing, plotting style, two-GPU job runner
task1/       config, data, methods (LoRA/OPLoRA/L2-SP), train, evaluate, qualitative, diagnostics, analyze, pipeline
task2/       config, data, features, model, train, evaluate, vqa_norm, analyze, pipeline
task3/       config, answers, data, selection, signals, adaptive, live, analyze
notebooks/   build_task*.py generate the guided Kaggle notebooks
tests/       tiny models, synthetic fixtures, unit tests
```

## Attribution

* LoRA/OPLoRA layers, the L2-SP penalty and all training loops were written for this assignment.
  LoRA initialisation follows PEFT's defaults (Hu et al., 2021; Hugging Face PEFT). OPLoRA's
  projection follows Xiong and Xie (2025).
* Benchmarks use EleutherAI's lm-evaluation-harness with its stock task definitions.
* `task2/vqa_norm.py` re-implements the answer normalisation and soft accuracy of the official VQA
  evaluation code (GT-vision-lab/VQA, `vqaEval.py`).
* `task3/answers.py` re-implements MathVista's `normalize_extracted_answer`
  (lupantech/MathVista, `evaluation/calculate_score.py`, commit c5acd37) and uses math-verify 0.9.0
  for MATH equivalence.
* The 24 qualitative prompts are `evaluation/first_plot_questions.yaml` from
  emergent-misalignment/emergent-misalignment (commit 80c1196).
* Code was written with assistance from an LLM (Claude), as permitted by the assignment's coding
  policy; every component is covered by the tests above.

## Data and checkpoints

Raw datasets, images, feature caches and large checkpoints are not committed. They are
downloaded from their official sources by the scripts above. Split and example IDs are saved
under `results/<task>/data/` (Tasks 1-2) and `results/task3/*/splits.json` (Task 3).
