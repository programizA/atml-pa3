"""Task 2 data: download COCO 2014 + VQAv2 annotations, then select every split with seed 6304.

    python -m task2.data download --raw /tmp/pa3_raw
    python -m task2.data select   --raw /tmp/pa3_raw --out results/task2/data

What gets selected (all IDs are saved to selection.json):

* COCO captions: 6,000 train2014 images x their first 3 captions (18,000 examples), plus
  96 val2014 development images x their first caption.
* Annotation-derived grounding questions, generated from COCO instance annotations:
    presence  "Does this image contain a dog?"  pair = an image with a clearly visible dog
              (>= 1% of the image) + an image with no dog annotation at all -> yes / no
    counting  "How many dogs can you see?"      pair = two images whose dog counts differ (1-4),
              counting only images where every dog is >= 0.5% of the image and none is a
              "crowd" region, so the count is unambiguous
  Training uses reusable pools built from the training images; development (24 pairs) and
  final test (64 pairs) come from val2014.
* Natural VQAv2 complementary pairs: the same question on two images whose consensus answers
  differ, each with >= 8 of 10 annotators agreeing. 1,536 training pairs (train2014), 32 dev and
  96 test pairs (val2014), split by answer type as the assignment specifies.

Every development and test image is used exactly once across all evaluation sets, so all
evaluation sets are image-disjoint from one another and (being val2014) from training.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.utils import read_json, write_json
from task2 import config as C
from task2.vqa_norm import consensus, normalize


# ------------------------------------------------------------------------------------- download
def _fetch(url, dest):
    import urllib.request

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print("downloading", url, flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def download(raw):
    raw = Path(raw)
    z = _fetch(C.COCO_ANN_URL, raw / "annotations_trainval2014.zip")
    with zipfile.ZipFile(z) as f:
        for name in ["annotations/captions_train2014.json", "annotations/captions_val2014.json",
                     "annotations/instances_train2014.json", "annotations/instances_val2014.json"]:
            if not (raw / name).exists():
                f.extract(name, raw)
    for key, (zname, member) in C.VQA_ZIPS.items():
        if (raw / "vqa" / member).exists():
            continue
        zp = _fetch(C.VQA_BASE + zname, raw / "vqa" / zname)
        with zipfile.ZipFile(zp) as f:
            names = f.namelist()
            m = next(n for n in names if n.endswith(member))
            with f.open(m) as src, open(raw / "vqa" / member, "wb") as dst:
                dst.write(src.read())
    print("annotations ready in", raw)


# ---------------------------------------------------------------------------------------- loading
def load_coco(raw, split):
    raw = Path(raw) / "annotations"
    caps = read_json(raw / f"captions_{split}.json")
    inst = read_json(raw / f"instances_{split}.json")
    images = {im["id"]: {"w": im["width"], "h": im["height"]} for im in inst["images"]}
    for im in caps["images"]:
        images.setdefault(im["id"], {"w": im["width"], "h": im["height"]})
    captions = defaultdict(list)
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        captions[a["image_id"]].append(a["caption"].strip())
    objects = defaultdict(list)
    for a in inst["annotations"]:
        objects[a["image_id"]].append((a["category_id"], float(a["area"]), int(a.get("iscrowd", 0))))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    return images, captions, objects, cats


def load_vqa(raw, split):
    v = Path(raw) / "vqa"
    tag = "train2014" if split == "train2014" else "val2014"
    qs = {q["question_id"]: q for q in read_json(v / f"v2_OpenEnded_mscoco_{tag}_questions.json")["questions"]}
    anns = {a["question_id"]: a for a in read_json(v / f"v2_mscoco_{tag}_annotations.json")["annotations"]}
    pairs = read_json(v / f"v2_mscoco_{tag}_complementary_pairs.json")
    return qs, anns, pairs


def key(split, image_id):
    return f"{split}/{int(image_id)}"


# -------------------------------------------------------------------------------------- selection
def vqa_candidates(qs, anns, pairs, split):
    """Complementary pairs that pass the agreement and different-answer filters."""
    out = []
    for q1, q2 in pairs:
        if q1 not in anns or q2 not in anns or q1 not in qs or q2 not in qs:
            continue
        a1, a2 = anns[q1], anns[q2]
        if a1["answer_type"] != a2["answer_type"] or a1["image_id"] == a2["image_id"]:
            continue
        if normalize(qs[q1]["question"]) != normalize(qs[q2]["question"]):
            continue
        r1 = [x["answer"] for x in a1["answers"]]
        r2 = [x["answer"] for x in a2["answers"]]
        c1, n1, t1 = consensus(r1)
        c2, n2, t2 = consensus(r2)
        if n1 < C.MIN_AGREE or n2 < C.MIN_AGREE or c1 == c2:
            continue
        out.append({"kind": "vqa", "answer_type": a1["answer_type"], "question": qs[q1]["question"],
                    "qid_a": q1, "qid_b": q2, "image_a": key(split, a1["image_id"]), "image_b": key(split, a2["image_id"]),
                    "answer_a": t1, "answer_b": t2, "canon_a": c1, "canon_b": c2, "refs_a": r1, "refs_b": r2,
                    "question_type": a1.get("question_type")})
    out.sort(key=lambda p: (p["qid_a"], p["qid_b"]))
    return out


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def plural(name):
    if name in C.PLURALS:
        return C.PLURALS[name]
    return name + ("es" if name.endswith(("s", "sh", "ch", "x")) else "s")


def object_tables(images, objects, cats, image_ids):
    """Per category: presence-positive images, images with no annotation of it, and valid counts."""
    pos, absent, counts = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for iid in sorted(image_ids):
        im = images.get(iid)
        if im is None:
            continue
        area = im["w"] * im["h"]
        per = defaultdict(list)
        for c, a, crowd in objects.get(iid, []):
            per[c].append((a / area, crowd))
        for c in cats:
            inst = per.get(c, [])
            if not inst:
                absent[c].append(iid)
                continue
            if any(fr >= C.PRESENCE_MIN_AREA for fr, crowd in inst if not crowd):
                pos[c].append(iid)
            if all(not crowd and fr >= C.COUNT_MIN_AREA for fr, crowd in inst):
                n = len(inst)
                if C.COUNT_RANGE[0] <= n <= C.COUNT_RANGE[1]:
                    counts[c][n].append(iid)
    return pos, absent, counts


def presence_pair(rng, c, cats, pos, absent, split, used=None):
    name = cats[c]
    cand_p = [i for i in pos[c] if used is None or key(split, i) not in used]
    cand_n = [i for i in absent[c] if used is None or key(split, i) not in used]
    if not cand_p or not cand_n:
        return None
    p, n = int(rng.choice(cand_p)), int(rng.choice(cand_n))
    return {"kind": "presence", "category": name, "question": f"Does this image contain {article(name)} {name}?",
            "image_a": key(split, p), "image_b": key(split, n), "answer_a": "yes", "answer_b": "no",
            "canon_a": "yes", "canon_b": "no"}


def counting_pair(rng, c, cats, counts, split, used=None):
    name = cats[c]
    avail = {n: [i for i in ims if used is None or key(split, i) not in used] for n, ims in counts[c].items()}
    avail = {n: v for n, v in avail.items() if v}
    if len(avail) < 2:
        return None
    n1, n2 = rng.choice(sorted(avail), size=2, replace=False)
    a, b = int(rng.choice(avail[n1])), int(rng.choice(avail[n2]))
    return {"kind": "counting", "category": name, "question": f"How many {plural(name)} can you see?",
            "image_a": key(split, a), "image_b": key(split, b), "answer_a": str(n1), "answer_b": str(n2),
            "canon_a": str(n1), "canon_b": str(n2)}


def select(raw, out, quick=False):
    rng = np.random.default_rng(C.SEED)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sel = {"seed": C.SEED, "notes": {}}
    used_eval = set()

    # ---- val2014: evaluation sets, image-disjoint across all of them
    v_images, v_caps, v_objs, cats = load_coco(raw, "val2014")
    qs, anns, pairs = load_vqa(raw, "val2014")
    vc = vqa_candidates(qs, anns, pairs, "val2014")
    sel["notes"]["vqa_val_candidates"] = {t: sum(p["answer_type"] == t for p in vc) for t in C.VQA_COUNTS["test"]}
    need = {s: dict(C.VQA_COUNTS[s]) for s in ("test", "dev")}
    vqa_eval = {"test": [], "dev": []}
    for i in rng.permutation(len(vc)):
        p = vc[i]
        if p["image_a"] in used_eval or p["image_b"] in used_eval:
            continue
        for s in ("test", "dev"):
            if need[s].get(p["answer_type"], 0) > 0:
                need[s][p["answer_type"]] -= 1
                vqa_eval[s].append(p)
                used_eval.update([p["image_a"], p["image_b"]])
                break
        if all(v == 0 for s in need for v in need[s].values()):
            break
    assert all(v == 0 for s in need for v in need[s].values()), f"not enough VQA val pairs: {need}"

    vpos, vabs, vcnt = object_tables(v_images, v_objs, cats, list(v_images))
    ground_eval = {}
    for s in ("test", "dev"):
        lst = []
        for kind, n in C.GROUND_EVAL[s].items():
            got, tries = 0, 0
            while got < n and tries < 10000:
                tries += 1
                c = int(rng.choice(sorted(cats)))
                pr = (presence_pair(rng, c, cats, vpos, vabs, "val2014", used_eval) if kind == "presence"
                      else counting_pair(rng, c, cats, vcnt, "val2014", used_eval))
                if pr is None:
                    continue
                used_eval.update([pr["image_a"], pr["image_b"]])
                lst.append(pr)
                got += 1
            assert got == n, f"could not build {n} {kind} pairs for {s}"
        ground_eval[s] = lst

    cap_dev_pool = sorted(i for i in v_caps if key("val2014", i) not in used_eval and len(v_caps[i]) >= 1)
    cap_dev = [{"kind": "caption", "image": key("val2014", int(i)), "target": v_caps[int(i)][0]}
               for i in rng.choice(cap_dev_pool, size=C.N_CAP_DEV, replace=False)]
    used_eval.update(x["image"] for x in cap_dev)
    del v_images, v_caps, v_objs, qs, anns, pairs

    # ---- train2014: caption examples, VQA training pairs, grounding pools
    t_images, t_caps, t_objs, _ = load_coco(raw, "train2014")
    cap_pool = sorted(i for i in t_caps if len(t_caps[i]) >= C.CAPS_PER_TRAIN_IMAGE)
    cap_imgs = sorted(int(i) for i in rng.choice(cap_pool, size=C.N_CAP_TRAIN_IMAGES, replace=False))
    cap_train = [{"kind": "caption", "image": key("train2014", i), "target": t_caps[i][j], "caption_index": j}
                 for i in cap_imgs for j in range(C.CAPS_PER_TRAIN_IMAGE)]
    qs, anns, pairs = load_vqa(raw, "train2014")
    tc = vqa_candidates(qs, anns, pairs, "train2014")
    sel["notes"]["vqa_train_candidates"] = {t: sum(p["answer_type"] == t for p in tc) for t in C.VQA_COUNTS["train"]}
    needt = dict(C.VQA_COUNTS["train"])
    vqa_train = []
    for i in rng.permutation(len(tc)):
        p = tc[i]
        if needt.get(p["answer_type"], 0) > 0:
            needt[p["answer_type"]] -= 1
            vqa_train.append(p)
        if all(v == 0 for v in needt.values()):
            break
    assert all(v == 0 for v in needt.values()), f"not enough VQA train pairs: {needt}"

    universe = sorted(set(cap_imgs) | {int(p[k].split("/")[1]) for p in vqa_train for k in ("image_a", "image_b")})
    tpos, tabs, tcnt = object_tables(t_images, t_objs, cats, universe)
    pres_pool, count_pool = [], []
    for c in sorted(cats):
        for _ in range(C.POOL_PAIRS_PER_CATEGORY if not quick else 5):
            pr = presence_pair(rng, c, cats, tpos, tabs, "train2014")
            if pr:
                pres_pool.append(pr)
            cp = counting_pair(rng, c, cats, tcnt, "train2014")
            if cp:
                count_pool.append(cp)
    # de-duplicate identical pairs (same images and category)
    pres_pool = list({(p["category"], p["image_a"], p["image_b"]): p for p in pres_pool}.values())
    count_pool = list({(p["category"], p["image_a"], p["image_b"]): p for p in count_pool}.values())

    for name, lst in [("vqa_train", vqa_train), ("vqa_dev", vqa_eval["dev"]), ("vqa_test", vqa_eval["test"]),
                      ("ground_dev", ground_eval["dev"]), ("ground_test", ground_eval["test"]),
                      ("presence_pool", pres_pool), ("counting_pool", count_pool)]:
        for j, p in enumerate(lst):
            p["pair_id"] = f"{name}:{j}"
    data = {"caption_train": cap_train, "caption_dev": cap_dev, "presence_pool": pres_pool,
            "counting_pool": count_pool, "vqa_train": vqa_train, "ground_dev": ground_eval["dev"],
            "ground_test": ground_eval["test"], "vqa_dev": vqa_eval["dev"], "vqa_test": vqa_eval["test"]}
    write_json(out / "selection.json", data, indent=None)

    images = sorted({x["image"] for x in cap_train + cap_dev}
                    | {p[k] for n in ("presence_pool", "counting_pool", "vqa_train", "ground_dev", "ground_test",
                                      "vqa_dev", "vqa_test") for p in data[n] for k in ("image_a", "image_b")})
    train_imgs = {i for i in images if i.startswith("train2014")}
    eval_imgs = {i for i in images if i.startswith("val2014")}
    sel.update({
        "sizes": {k: len(v) for k, v in data.items()},
        "n_images": len(images), "n_train_images": len(train_imgs), "n_eval_images": len(eval_imgs),
        "pool_categories": {"presence": len({p["category"] for p in pres_pool}),
                            "counting": len({p["category"] for p in count_pool})},
        "counting_answer_distribution": {s: _dist([p[k] for p in data[s] for k in ("answer_a", "answer_b")])
                                         for s in ("counting_pool",)} | {
            "ground_test_counting": _dist([p[k] for p in data["ground_test"] if p["kind"] == "counting"
                                           for k in ("answer_a", "answer_b")])},
        "isolation": isolation_report(data),
    })
    write_json(out / "selection_summary.json", sel)
    (out / "images.txt").write_text("\n".join(images) + "\n")
    write_json(out / "ids.json", {
        "caption_train_images": cap_imgs, "caption_dev_images": [x["image"] for x in cap_dev],
        **{f"{n}_pairs": [{k: p.get(k) for k in ("pair_id", "qid_a", "qid_b", "image_a", "image_b", "category", "kind",
                                                 "answer_type")} for p in data[n]]
           for n in ("vqa_train", "vqa_dev", "vqa_test", "ground_dev", "ground_test")}})
    return sel


def _dist(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def isolation_report(data):
    """Check the split rules: evaluation images never in training/replay; eval sets disjoint."""
    def imgs(name):
        lst = data[name]
        if name.startswith("caption"):
            return {x["image"] for x in lst}
        return {p[k] for p in lst for k in ("image_a", "image_b")}
    train = imgs("caption_train") | imgs("presence_pool") | imgs("counting_pool") | imgs("vqa_train")
    evals = ["caption_dev", "ground_dev", "ground_test", "vqa_dev", "vqa_test"]
    rep = {"train_eval_overlap": {e: len(train & imgs(e)) for e in evals}, "eval_eval_overlap": {}}
    for i, a in enumerate(evals):
        for b in evals[i + 1:]:
            rep["eval_eval_overlap"][f"{a}~{b}"] = len(imgs(a) & imgs(b))
    rep["within_eval_reuse"] = {e: (2 * len(data[e]) - len(imgs(e))) if not e.startswith("caption") else
                                len(data[e]) - len(imgs(e)) for e in evals}
    rep["ok"] = (all(v == 0 for v in rep["train_eval_overlap"].values())
                 and all(v == 0 for v in rep["eval_eval_overlap"].values())
                 and all(v == 0 for v in rep["within_eval_reuse"].values()))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "select"])
    ap.add_argument("--raw", default="/tmp/pa3_raw")
    ap.add_argument("--out", default="results/task2/data")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.raw)
    else:
        print(json.dumps(select(a.raw, a.out, a.quick), indent=1))
