"""Small synthetic stand-ins for the public datasets, shaped exactly like the real rows."""
from __future__ import annotations

import json
import random
from pathlib import Path


def make_text_fixtures(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)
    gsm_test = [{"question": f"Tom has {i} marbles and finds {i+2} more. How many marbles now?",
                 "answer": f"He has {i} + {i+2} = {2*i+2}.\n#### {2*i+2}"} for i in range(40)]
    rows = []
    for i in range(160):
        a, b = rnd.randint(1, 50), rnd.randint(1, 50)
        rows.append({"query": f"Sara has {a} apples and buys {b} more. How many apples does she have?",
                     "response": f"Sara starts with {a}. She buys {b}. {a} + {b} = {a+b}. The answer is: {a+b}",
                     "type": rnd.choice(["GSM_Rephrased", "MATH_AnsAug", "GSM_SV"]),
                     "original_question": f"orig {i}"})
    # the three exclusion rules, each triggered once inside the selected range
    rows[3]["response"] = "   "
    rows[7]["original_question"] = gsm_test[5]["question"].upper() + "  "   # normalisation match
    rows[11]["response"] = "word " * 3000                                      # > 2048 tokens
    hs = [{"ind": i, "activity_label": "Roof shingle removal",
           "ctx_a": "A man is sitting on a roof.", "ctx_b": "he",
           "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
                       "is holding a rubik's cube.", "starts pulling up roofing on a roof."],
           "label": "3"} for i in range(50)]
    for name, obj in [("metamath.json", rows), ("gsm8k_test.json", gsm_test), ("hellaswag_val.json", hs)]:
        with open(root / name, "w") as f:
            json.dump(obj, f)
    return root


if __name__ == "__main__":
    print(make_text_fixtures("/tmp/fx"))
