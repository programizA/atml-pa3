"""VQA answer normalisation and soft accuracy, following the official VQA evaluation code
(github.com/GT-vision-lab/VQA, vqaEval.py). The same normaliser is applied to every answer string:
training targets, human references, and model predictions.

Steps: strip whitespace and newlines; handle punctuation (removed when it touches a space or sits
in a number like 1,000, otherwise replaced by a space); drop periods that are not decimal points;
lower-case; number words -> digits ("two" -> "2"); drop articles (a, an, the); fix common
contractions ("dont" -> "don't").
"""
from __future__ import annotations

import re

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
           "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_period = re.compile(r"(?!<=\d)(\.)(?!\d)")
_comma = re.compile(r"(\d)(,)(\d)")


def _punct(t: str) -> str:
    out = t
    for p in PUNCT:
        if (p + " " in t or " " + p in t) or _comma.search(t) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _period.sub("", out)


def _digits_articles(t: str) -> str:
    words = [NUMBERS.get(w, w) for w in t.lower().split()]
    words = [w for w in words if w not in ARTICLES]
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize(ans: str) -> str:
    if ans is None:
        return ""
    t = ans.replace("\n", " ").replace("\t", " ").strip()
    return _digits_articles(_punct(t))


def soft_accuracy(pred: str, refs: list[str]) -> float:
    """s(a) = (1/10) * sum_i min(1, (1/3) * #{j != i : a == a_j}) with normalised strings.
    Averaging over the ten leave-one-out subsets is the official definition."""
    p = normalize(pred)
    r = [normalize(x) for x in refs]
    n = len(r)
    tot = 0.0
    for i in range(n):
        match = sum(1 for j in range(n) if j != i and r[j] == p)
        tot += min(1.0, match / 3.0)
    return tot / n


def consensus(refs: list[str]):
    """(most common normalised answer, its count, the original string chosen as training target).
    The target is the most frequent raw spelling inside the consensus group (ties: first seen)."""
    norm = [normalize(x) for x in refs]
    counts = {}
    for a in norm:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts, key=lambda a: (counts[a], -norm.index(a)))
    raw = [x.strip() for x, n in zip(refs, norm) if n == best]
    rc = {}
    for x in raw:
        rc[x] = rc.get(x, 0) + 1
    target = max(rc, key=lambda x: (rc[x], -raw.index(x)))
    return best, counts[best], target
