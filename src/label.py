"""Build the golden evaluation set.

  propose    stratified sample + model-proposed labels
  goldlabel  AI annotation pass (pro) + inter-annotator reliability (flash)
  review     human accepts/corrects each (the true gold step)
  status     distribution + how many are verified

Labels are proposed with a different model (pro) than the classifier under test
(flash) to avoid baking the classifier's own answers into its answer key.
"""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from src.common import config, gemini_client, ROOT, generate_json
from src.classify import classify
from src.escalate import decide
from src.retrieve import retrieve
from src.taxonomy import LABELS, taxonomy_prompt_block

TARGET = 220          # aim for the top of the 150-250 band
FLOOR = 12            # try to get at least this many of each intent
BUDGET = 300          # how many pool rows to run the proposer over


def _pool():
    cfg = config()
    path = ROOT / cfg["data"]["sample_dir"] / "eval_pool.jsonl"
    return [json.loads(l) for l in open(path)]


def propose():
    cfg = config()
    client = gemini_client()
    pool = _pool()
    random.seed(cfg["data"]["random_seed"])
    random.shuffle(pool)

    retrieve("warm up the cached index", client=client)   # build index single-threaded

    def propose_one(row):
        intent = classify(row["customer_text"], client=client,
                          model=cfg["models"]["judge"])    # strong model proposes
        retrieved = retrieve(row["customer_text"], client=client)
        dec = decide(row["customer_text"], intent, retrieved)
        return {
            "cust_tweet_id": row["cust_tweet_id"],
            "customer_text": row["customer_text"],
            "reference_reply": row["brand_reply"],   # what Amazon actually sent
            "proposed_intent": intent["intent"],
            "proposed_decision": dec["decision"],
            "intent": intent["intent"],              # editable final fields
            "decision": dec["decision"],
            "verified": False,
            "difficulty": None,
        }

    buckets = {k: [] for k in LABELS}
    with ThreadPoolExecutor(max_workers=8) as ex:
        proposed = list(ex.map(propose_one, pool[:BUDGET]))
    for rec in proposed:
        buckets[rec["proposed_intent"]].append(rec)

    # Stratified trim toward TARGET while respecting the per-class floor.
    picked = []
    for k in LABELS:
        picked.extend(buckets[k][:max(FLOOR, 0)])
    extra = [r for r in proposed if r not in picked]
    random.shuffle(extra)
    picked.extend(extra[:max(0, TARGET - len(picked))])

    out = ROOT / "golden" / "golden_draft.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = Counter(r["proposed_intent"] for r in picked)
    print(f"Proposed {len(picked)} examples -> {out}")
    for k in LABELS:
        print(f"  {k:20} {dist.get(k, 0)}")
    print("\nNow run: python -m src.label review")


class GoldLabel(BaseModel):
    intent: str
    decision: str      # "auto" or "escalate"
    difficulty: str    # "easy" | "med" | "hard"
    rationale: str


GOLD_PROMPT = """You are an expert annotator building the gold evaluation set for
an AmazonHelp support agent. Label this customer message independently and
carefully. You are shown the reply Amazon actually sent, as evidence of how the
issue was really handled; use it to inform your labels but label what is correct,
not just what Amazon did.

Intents (choose exactly one):
{taxonomy}

decision = should an automated agent handle this itself, or escalate to a human?
- auto: the agent can post a complete, helpful public reply (answer a question,
  give troubleshooting or standard next steps) with no human needed.
- escalate: a human must take over, because it needs account-specific action,
  involves money/refunds, security or fraud, legal or sensitive matters, strong
  complaint/anger, or there is no safe standard reply. If Amazon's real reply had
  to route the customer to a DM, phone line, form, or specialist, that is usually
  a sign the issue needed a human: lean escalate.

difficulty = how hard is this to label/handle: easy, med, or hard.

Customer message:
\"\"\"{message}\"\"\"

What Amazon actually replied:
\"\"\"{reference}\"\"\"
"""


def _gold_one(row, client, model):
    prompt = GOLD_PROMPT.format(taxonomy=taxonomy_prompt_block(),
                                message=row["customer_text"],
                                reference=row["reference_reply"])
    out = generate_json(client, model, prompt, GoldLabel)
    intent = out.get("intent") if out.get("intent") in LABELS else "general_other"
    decision = "escalate" if out.get("decision") == "escalate" else "auto"
    difficulty = out.get("difficulty") if out.get("difficulty") in \
        ("easy", "med", "hard") else "med"
    return intent, decision, difficulty, out.get("rationale", "")


def goldlabel():
    """AI annotation pass. Labels all draft rows with the strong `pro` model,
    independently of the rule gate, and records a second model's labels so we can
    report inter-annotator agreement (a measure of label reliability)."""
    cfg = config()
    client = gemini_client()
    draft = ROOT / "golden" / "golden_draft.jsonl"
    rows = [json.loads(l) for l in open(draft)]

    primary = cfg["models"]["judge"]          # pro: the gold annotator
    secondary = cfg["models"]["classify"]     # flash: reliability cross-check

    def annotate(model):
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(lambda r: _gold_one(r, client, model), rows))

    print(f"Annotating {len(rows)} rows with {primary} (gold) ...")
    gold = annotate(primary)
    print(f"Cross-annotating with {secondary} (reliability) ...")
    cross = annotate(secondary)

    for r, (intent, decision, diff, why) in zip(rows, gold):
        r["intent"], r["decision"], r["difficulty"] = intent, decision, diff
        r["label_rationale"] = why
        r["verified"] = True
        r["annotator"] = primary

    gpath = ROOT / "golden" / "golden_set.jsonl"
    with open(gpath, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Inter-annotator agreement (pro vs flash) on intent and decision.
    from sklearn.metrics import cohen_kappa_score
    gi, ci = [g[0] for g in gold], [c[0] for c in cross]
    gd, cd = [g[1] for g in gold], [c[1] for c in cross]
    iaa = {
        "intent_cohen_kappa": round(cohen_kappa_score(gi, ci), 4),
        "intent_exact": round(sum(a == b for a, b in zip(gi, ci)) / len(gi), 4),
        "decision_cohen_kappa": round(cohen_kappa_score(gd, cd), 4),
        "decision_exact": round(sum(a == b for a, b in zip(gd, cd)) / len(gd), 4),
        "primary": primary, "secondary": secondary, "n": len(rows),
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "inter_annotator.json", "w") as f:
        json.dump(iaa, f, indent=2)

    print(f"\nWrote {gpath}")
    print("Inter-annotator agreement (label reliability):")
    print(f"  intent   kappa={iaa['intent_cohen_kappa']}  exact={iaa['intent_exact']}")
    print(f"  decision kappa={iaa['decision_cohen_kappa']}  exact={iaa['decision_exact']}")
    print("Distribution:", dict(Counter(r["intent"] for r in rows)))
    print("Decisions:", dict(Counter(r["decision"] for r in rows)))


def review():
    draft = ROOT / "golden" / "golden_draft.jsonl"
    gold = ROOT / "golden" / "golden_set.jsonl"
    rows = [json.loads(l) for l in open(draft)]
    # Resume: keep any already-verified rows from a prior session.
    done = {}
    if gold.exists():
        for l in open(gold):
            r = json.loads(l)
            done[r["cust_tweet_id"]] = r

    menu = "  ".join(f"{i}={k}" for i, k in enumerate(LABELS))
    total = len(rows)
    print("\nReview golden labels. ENTER accepts the proposal as-is.")
    print("To correct, type tokens separated by spaces, any order:")
    print("  an intent number (0-7), 'a'/'e' for the decision, easy/med/hard.")
    print("  e.g. '3'  or  '3 e'  or  'e hard'.  'q' = save and quit.\n")
    print(menu, "\n")

    n = 0
    for idx, r in enumerate(rows):
        if r["cust_tweet_id"] in done and done[r["cust_tweet_id"]]["verified"]:
            continue
        print("-" * 70)
        print(f"[{idx+1}/{total}]  MSG :", r["customer_text"])
        print("          REF :", r["reference_reply"][:140])
        print(f"          proposed: intent={r['proposed_intent']}  "
              f"decision={r['proposed_decision']}")
        ans = input("          accept[enter] / correct / q: ").strip().lower()
        if ans == "q":
            break
        r["difficulty"] = "med"
        for tok in ans.split():
            if tok.isdigit() and int(tok) < len(LABELS):
                r["intent"] = LABELS[int(tok)]
            elif tok == "a":
                r["decision"] = "auto"
            elif tok == "e":
                r["decision"] = "escalate"
            elif tok in ("easy", "med", "hard"):
                r["difficulty"] = tok
        r["verified"] = True
        done[r["cust_tweet_id"]] = r
        n += 1

    with open(gold, "w") as f:
        for r in rows:
            f.write(json.dumps(done.get(r["cust_tweet_id"], r), ensure_ascii=False) + "\n")
    verified = sum(1 for r in done.values() if r.get("verified"))
    print(f"\nSaved {gold}. Verified this session: {n}. Total verified: {verified}")


def status():
    gold = ROOT / "golden" / "golden_set.jsonl"
    src = gold if gold.exists() else ROOT / "golden" / "golden_draft.jsonl"
    rows = [json.loads(l) for l in open(src)]
    verified = sum(1 for r in rows if r.get("verified"))
    print(f"File: {src.name}  rows: {len(rows)}  human-verified: {verified}")
    print("Intent distribution:", dict(Counter(r["intent"] for r in rows)))
    print("Decision distribution:", dict(Counter(r["decision"] for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["propose", "goldlabel", "review", "status"])
    a = ap.parse_args()
    {"propose": propose, "goldlabel": goldlabel, "review": review,
     "status": status}[a.cmd]()
