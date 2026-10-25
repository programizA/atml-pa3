"""Mini COCO-2014 / VQAv2 files in the official formats, plus small JPEGs, for CPU tests."""
import json, random, sys
from pathlib import Path
from PIL import Image

CATS = [{"id": i + 1, "name": n, "supercategory": "x"} for i, n in enumerate(["dog", "cat", "person", "apple", "bus"])]
ANS = {"yes/no": ["yes", "no"], "number": ["1", "2", "3", "4"], "other": ["red", "blue", "green", "white"]}


def build(root, n_train=120, n_val=200, seed=0):
    root = Path(root); rnd = random.Random(seed)
    (root / "annotations").mkdir(parents=True, exist_ok=True); (root / "vqa").mkdir(parents=True, exist_ok=True)
    qid = 1
    for split, n in [("train2014", n_train), ("val2014", n_val)]:
        img_dir = root / "images" / split; img_dir.mkdir(parents=True, exist_ok=True)
        ids = list(range(1000, 1000 + n))
        images = [{"id": i, "width": 64, "height": 48, "file_name": f"COCO_{split}_{i:012d}.jpg"} for i in ids]
        caps, inst, aid = [], [], 1
        for i in ids:
            Image.new("RGB", (64, 48), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))).save(img_dir / f"COCO_{split}_{i:012d}.jpg")
            for j in range(5):
                caps.append({"id": aid, "image_id": i, "caption": f"A photo number {i} with thing {j}."}); aid += 1
            for c in rnd.sample(CATS, 2):
                for _ in range(rnd.randint(1, 4)):
                    inst.append({"id": aid, "image_id": i, "category_id": c["id"], "area": rnd.choice([200.0, 900.0]),
                                 "iscrowd": int(rnd.random() < 0.05), "bbox": [0, 0, 10, 10]}); aid += 1
        json.dump({"images": images, "annotations": caps}, open(root / "annotations" / f"captions_{split}.json", "w"))
        json.dump({"images": images, "annotations": inst, "categories": CATS}, open(root / "annotations" / f"instances_{split}.json", "w"))
        qs, anns, pairs = [], [], []
        for k in range(n // 2 if split == "val2014" else n):
            a, b = rnd.sample(ids, 2) if split == "train2014" else (ids[2 * k], ids[2 * k + 1])
            t = rnd.choice(list(ANS)); x, y = rnd.sample(ANS[t], 2)
            q = f"What is shown {t} {k}?"
            for img, ans in [(a, x), (b, y)]:
                refs = [ans] * rnd.choice([8, 9, 10]); refs += [rnd.choice(ANS[t])] * (10 - len(refs))
                qs.append({"question_id": qid, "image_id": img, "question": q})
                anns.append({"question_id": qid, "image_id": img, "answer_type": t, "question_type": "what",
                             "multiple_choice_answer": ans, "answers": [{"answer": r, "answer_id": j + 1} for j, r in enumerate(refs)]})
                qid += 1
            pairs.append([qid - 2, qid - 1])
        json.dump({"questions": qs}, open(root / "vqa" / f"v2_OpenEnded_mscoco_{split}_questions.json", "w"))
        json.dump({"annotations": anns}, open(root / "vqa" / f"v2_mscoco_{split}_annotations.json", "w"))
        json.dump(pairs, open(root / "vqa" / f"v2_mscoco_{split}_complementary_pairs.json", "w"))
    return root


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fx2"))
