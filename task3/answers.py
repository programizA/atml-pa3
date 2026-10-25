"""Answer extraction, normalisation, grouping and grading for Task 3.

Grouping decides which rollouts "agree" (used for voting); grading decides which group is
correct. Both use the same equivalence rule and never look at the gold answer when forming
groups. Missing extractions form one NO_ANSWER group, which is always graded incorrect.

MATH        math-verify (fixed version, see SCORER_VERSION): two answers are equivalent when
            math-verify judges them mathematically equal (e.g. \\frac{1}{2} == 0.5,
            10{,}000 == 10000). Each question's distinct strings are grouped greedily in rollout
            order; a group is graded by math-verify against the gold answer.
MathVista   the official MathVista rules (evaluation/calculate_score.py), applied after a light
            LaTeX clean-up so that "51.045\\%" or "118.6 \\, \\text{min}" can be read as numbers:
            multi-choice -> option letter ("(B)" or "B" directly; otherwise the closest choice by
            edit distance, as MathVista does); integer -> int(float(x)); float -> rounded to the
            question's precision; list -> the string itself. Numbers written in LaTeX that plain
            float() cannot read (\\frac{3}{14}, 5\\sqrt{2}) are evaluated with math-verify. A
            non-empty answer that still cannot be read becomes its own INVALID group (graded wrong).
"""
from __future__ import annotations

import re
from functools import lru_cache

NO_ANSWER = "NO_ANSWER"


def missing(x) -> bool:
    """None, NaN (pandas' stand-in for null) or an empty string."""
    return x is None or (isinstance(x, float) and x != x) or not str(x).strip()


def scorer_version():
    import math_verify

    try:
        from importlib.metadata import version

        return {"math_verify": version("math-verify"), "mathvista_rules": "calculate_score.py@c5acd37"}
    except Exception:
        return {"math_verify": getattr(math_verify, "__version__", "?")}


# ------------------------------------------------------------------------------ boxed extraction
def extract_boxed(text: str):
    """Last literal \\boxed{ ... } with balanced braces (README rule). Absent, empty or unclosed
    last box -> None. No numeric fallback."""
    if text is None:
        return None
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth, k = 1, j
    while k < len(text):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                ans = text[j:k].strip()
                return ans or None
        k += 1
    return None


# ------------------------------------------------------------------------------------------ MATH
@lru_cache(maxsize=200_000)
def _mv_parse(s: str):
    from math_verify import LatexExtractionConfig, parse

    try:
        return tuple(parse("\\boxed{" + s + "}", extraction_config=[LatexExtractionConfig()], parsing_timeout=5))
    except Exception:
        return tuple()


def _mv_equal(a, b) -> bool:
    from math_verify import verify

    if not a or not b:
        return False
    try:
        return bool(verify(list(a), list(b), timeout_seconds=5))
    except Exception:
        return False


def group_math(answers):
    """answers: list of extracted strings/None for one question, in rollout-id order.
    Returns (group label per answer, {label: representative string})."""
    labels, reps = [], {}
    rep_parsed = []          # (label, parsed)
    cache = {}
    for a in answers:
        if missing(a):
            labels.append(NO_ANSWER)
            continue
        s = str(a).strip()
        if s in cache:
            labels.append(cache[s])
            continue
        p = _mv_parse(s)
        lab = None
        if p:
            for gl, gp in rep_parsed:
                if _mv_equal(gp, p):
                    lab = gl
                    break
        else:
            lab = next((gl for gl, r in reps.items() if r == s), None)
        if lab is None:
            lab = f"g{len(reps)}"
            reps[lab] = s
            if p:
                rep_parsed.append((lab, p))
        cache[s] = lab
        labels.append(lab)
    return labels, reps


def grade_math(gold: str, rep: str) -> bool:
    if rep is None:
        return False
    g, p = _mv_parse(str(gold).strip()), _mv_parse(str(rep).strip())
    if g and p:
        return _mv_equal(g, p)
    return str(gold).strip() == str(rep).strip()


# ------------------------------------------------------------------------------------- MathVista
def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _clean_number(s: str) -> str:
    t = s.strip().strip("$")
    t = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{[^{}]*\}", " ", t)
    t = t.replace("\\%", "").replace("%", "").replace("{,}", "")
    t = re.sub(r"\\[,;:! ]", " ", t).replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\^\s*\{?\\circ\}?|\\circ|°|\\degree", "", t)
    t = re.sub(r"\\dfrac|\\tfrac", r"\\frac", t)
    t = t.strip()
    prev = None
    while prev != t:                     # leftovers of removed units: "10 / ^2" -> "10"
        prev = t
        t = re.sub(r"\s*(?:[/^]\s*\{?\d*\}?|~)\s*$", "", t).strip()
    m = re.match(r"^([-+]?[\d.,]+)\s*[a-zA-Z/^\d\s]*$", t)       # "235V", "12 cm", "8 m/s^2"
    if m and re.search(r"[a-zA-Z]", t):
        t = m.group(1)
    return t.strip()


def _to_float(s: str):
    t = _clean_number(s)
    try:
        return float(t.replace(",", "")) if re.fullmatch(r"[-+]?[\d,]*\.?\d+(e[-+]?\d+)?", t.replace(" ", "")) else float(t)
    except ValueError:
        pass
    p = _mv_parse(t)
    for x in p:
        try:
            import sympy

            if hasattr(x, "rhs"):
                x = x.rhs
            if isinstance(x, sympy.Basic) and x.is_number:
                return float(sympy.N(x))
        except Exception:
            continue
    return None


def normalize_mathvista(extraction, meta):
    """MathVista's normalize_extracted_answer, returning option letters for multi-choice."""
    if missing(extraction):
        return NO_ANSWER
    ext = str(extraction).strip()
    if meta["question_type"] == "multi_choice":
        choices = meta["choices"]
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        found = re.findall(r"\(([a-zA-Z])\)", ext)
        if found:
            ext = found[0].upper()
        if ext in letters:
            return ext
        dists = [_lev(ext, c) for c in choices]
        return letters[dists.index(min(dists))]
    at = meta["answer_type"]
    if at in ("integer", "float"):
        v = _to_float(ext)
        if v is None:
            return "INVALID:" + ext
        try:
            return str(int(v)) if at == "integer" else str(round(v, int(meta["precision"])))
        except (OverflowError, ValueError):
            return "INVALID:" + ext
    return str(ext)       # list


def gold_mathvista(gold, meta):
    if meta["question_type"] == "multi_choice":
        g = str(gold).strip()
        letters = [chr(ord("A") + i) for i in range(len(meta["choices"]))]
        return g if g in letters else letters[meta["choices"].index(meta["answer"])]
    return normalize_mathvista(gold, meta)
