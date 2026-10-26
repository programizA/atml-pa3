"""Build notebooks/task2_figures.ipynb (CPU only).  Run: python notebooks/build_task2_figures.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import code, md, save, setup_cell  # noqa: E402

cells = [
md("""
# Task 2 figures and label audit (CPU only, ~5 minutes)

Attach the **output of the finished Task 2 notebook** as an input (Add Input → Your Work). Its
`results/task2` folder is restored automatically. No GPU is needed: set Accelerator to **None**
and Internet to **on**.

1. **Label audit sheets:** every annotation-derived test pair, with its generated answers and no
   model predictions. Open the PNGs, and in `label_audit.csv` put `y` in the `doubtful(y/n)` column
   for every image whose generated answer you think is wrong or ambiguous. Pairs are numbered as
   on the sheets.
2. **Aligned examples figure** for the report.
"""),
setup_cell("task2"),
code('''
IMAGES = os.environ.get("PA3_FX2_IMAGES", "/tmp/pa3_images")
sh(f"python -m task2.figures --root {ROOT} --images {IMAGES}")
from IPython.display import Image, display
for f in sorted(glob.glob(f"{ROOT}/data/label_audit_*.png")):
    print(f); display(Image(f, width=1100))
'''),
code('''
for run in ["run", "run_pool49"]:
    if os.path.exists(f"{ROOT}/{run}/test_predictions.jsonl"):
        sh(f"python -m task2.analyze --root {ROOT} --run {ROOT}/{run} --images {IMAGES} > /dev/null")
display(Image(f"{ROOT}/run/report/fig_examples.png", width=900))
'''),
code('''
import zipfile
zp = f"{OUT}/task2_figures.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in list(pathlib.Path(ROOT, "data").glob("label_audit*")) + list(pathlib.Path(ROOT).glob("run*/report/*")):
        z.write(p, p.relative_to(OUT))
print(zp, round(os.path.getsize(zp) / 2**20, 1), "MB")
'''),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "task2_figures.ipynb"
    save(cells, out)
    print("wrote", out)
