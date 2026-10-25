"""Shared pieces for building the guided Kaggle notebooks (one per task)."""
from __future__ import annotations

import nbformat as nbf


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def save(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    nbf.write(nb, path)


def setup_cell(task):
    """Clone the repo, install pinned packages, restore results from earlier notebook versions
    attached as inputs, and define the paths every later cell uses. PA3_TEST=1 switches to the
    tiny CPU stand-ins used for testing the notebook itself."""
    return code(f'''
# ---- EDIT THIS: your public GitHub repository ----
REPO_URL = "https://github.com/<your-username>/<your-repo>.git"
BRANCH = "main"

import os, sys, glob, shutil, subprocess, time, json, pathlib
NOTEBOOK_START = time.time()
TEST = os.environ.get("PA3_TEST") == "1"          # only used when testing this notebook on a CPU
if TEST:
    REPO = os.environ["PA3_REPO"]
    OUT = os.environ.get("PA3_OUT", "/tmp/pa3_nbtest")
else:
    REPO = "/tmp/atml-pa3"                          # code: scratch disk, not saved
    OUT = "/kaggle/working"                         # results: saved with the notebook version
    os.environ["HF_HOME"] = "/tmp/hf"               # model/dataset caches: scratch, not saved
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "pull", "-q"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{{REPO}}/requirements.txt"], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
ROOT = f"{{OUT}}/results/{task}"
os.makedirs(ROOT, exist_ok=True)

# Resume support: attach an earlier version of this notebook (its Output) as an input dataset and
# everything it finished is copied back here, so finished jobs are skipped.
for prev in glob.glob(f"/kaggle/input/*/results/{task}") + glob.glob(f"/kaggle/input/*/*/results/{task}"):
    print("restoring finished work from", prev)
    shutil.copytree(prev, ROOT, dirs_exist_ok=True)

def sh(cmd):
    """Run a shell command from the repo, streaming its output; stop the notebook if it fails."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
    for line in p.stdout:
        if "Warning" not in line and "warn(" not in line:
            print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"command failed: {{cmd}}")

import torch
N_GPUS = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)] or "none (CPU test mode)")
print("commit:", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
''')
