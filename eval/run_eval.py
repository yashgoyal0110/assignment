"""Evaluation harness. Produces the headline numbers and writes outputs/.

  python -m eval.run_eval                 # full eval
  python -m eval.run_eval --reply-n 60
  python -m eval.run_eval --classify-only
  python -m eval.run_eval --compare-models gemini-3.5-flash gemini-3.5-flash-lite

Classification uses stratified 5-fold CV so no baseline is scored on its own
training data; the LLM classifier is zero-shot and evaluated out-of-fold.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.common import config, gemini_client, ROOT
from src.classify import classify
from src.retrieve import retrieve
from src.escalate import decide
from src.draft import draft
from src import baselines as B
from src.taxonomy import LABELS
from eval import metrics as M
from eval.judge import judge_reply

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_golden():
    gold = ROOT / "golden" / "golden_set.jsonl"
    draft_path = ROOT / "golden" / "golden_draft.jsonl"
    path = gold if gold.exists() else draft_path
    rows = [json.loads(l) for l in open(path)]
    verified = sum(1 for r in rows if r.get("verified"))
    print(f"Golden: {path.name}  n={len(rows)}  human-verified={verified}")
    if verified < len(rows):
        print("  WARNING: not all labels are human-verified. Numbers are "
              "provisional until `python -m src.label review` is complete.\n")
    return rows


def llm_classify_all(rows, model, client, tag):
    """Classify every golden message, cached per (model, tweet_id)."""
    cache = OUT / f"preds_{tag}.jsonl"
    done = {}
    if cache.exists():
        for l in open(cache):
            d = json.loads(l)
            done[d["id"]] = d["intent"]

    todo = [r for r in rows if r["cust_tweet_id"] not in done]
    if todo:
        def one(r):
            return r["cust_tweet_id"], classify(r["customer_text"], client=client,
                                                model=model)["intent"]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for tid, intent in ex.map(one, todo):
                done[tid] = intent
        with open(cache, "w") as f:
            for tid, intent in done.items():
                f.write(json.dumps({"id": tid, "intent": intent}) + "\n")
    return [done[r["cust_tweet_id"]] for r in rows]


def eval_classification(rows, client, model=None):
    cfg = config()
    model = model or cfg["models"]["classify"]
    X = [r["customer_text"] for r in rows]
    y = [r["intent"] for r in rows]
    tag = model.replace("/", "_")
    llm_pred = llm_classify_all(rows, model, client, tag)

    # Out-of-fold predictions for the two baselines.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    triv = [None] * len(rows)
    simp = [None] * len(rows)
    y_arr = np.array(y)
    for tr, te in skf.split(X, y):
        ytr = y_arr[tr]
        for i in te:
            triv[i] = Counter(ytr).most_common(1)[0][0]
        clf = B.simple_classifier()
        clf.fit([X[i] for i in tr], [y[i] for i in tr])
        preds = clf.predict([X[i] for i in te])
        for k, i in enumerate(te):
            simp[i] = preds[k]

    present = sorted(set(y))
    result = {
        "model": model,
        "trivial_majority": M.classification_scores(y, triv, present),
        "simple_tfidf_logreg": M.classification_scores(y, simp, present),
        "llm": M.classification_scores(y, llm_pred, present),
    }
    print("\n=== INTENT CLASSIFICATION (n={}) ===".format(len(rows)))
    for name in ["trivial_majority", "simple_tfidf_logreg", "llm"]:
        s = result[name]
        print(f"  {name:22} acc={s['accuracy']:.3f}  macroF1={s['macro_f1']:.3f}")
    print("\n  LLM per-class:\n" + M.per_class_report(y, llm_pred, present))
    result["confusion_labels"] = present
    result["confusion"] = M.confusion(y, llm_pred, present).tolist()
    return result, llm_pred


def eval_escalation(rows, client):
    """Agent gate vs human decision. Baseline: always-auto (do nothing)."""
    y = [r["decision"] for r in rows]

    def one(r):
        intent = classify(r["customer_text"], client=client)
        retrieved = retrieve(r["customer_text"], client=client)
        return decide(r["customer_text"], intent, retrieved)["decision"]

    with ThreadPoolExecutor(max_workers=8) as ex:
        pred = list(ex.map(one, rows))

    trivial = ["auto"] * len(rows)   # the do-nothing baseline
    result = {
        "trivial_always_auto": M.escalation_scores(y, trivial),
        "agent_gate": M.escalation_scores(y, pred),
    }
    print("\n=== ESCALATION DECISION (n={}, positive=escalate) ===".format(len(rows)))
    for name in ["trivial_always_auto", "agent_gate"]:
        s = result[name]
        print(f"  {name:20} acc={s['accuracy']:.3f}  "
              f"P={s['escalate_precision']:.3f} R={s['escalate_recall']:.3f} "
              f"F1={s['escalate_f1']:.3f}")
    return result, pred


def eval_replies(rows, client, n):
    cfg = config()
    subset = rows[:n]
    print(f"\n=== REPLY QUALITY (judged n={len(subset)}) ===")

    def build(r):
        retrieved = retrieve(r["customer_text"], client=client)
        intent = classify(r["customer_text"], client=client)["intent"]
        agent_reply = draft(r["customer_text"], intent, retrieved, client=client)
        return {
            "id": r["cust_tweet_id"], "message": r["customer_text"],
            "retrieved": retrieved,
            "replies": {
                "trivial": B.trivial_reply(r["customer_text"]),
                "simple": B.simple_reply(r["customer_text"], retrieved),
                "agent": agent_reply,
            },
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        built = list(ex.map(build, subset))

    judge_model = cfg["models"]["judge"]
    cross_model = cfg["models"]["judge_cross"]

    def judge_all(item):
        scores = {}
        for sys_name, reply in item["replies"].items():
            scores[sys_name] = judge_reply(item["message"], reply,
                                            item["retrieved"], model=judge_model,
                                            client=client)
        # cross judge only the agent reply, to check self-preference bias
        scores["agent_cross"] = judge_reply(item["message"], item["replies"]["agent"],
                                            item["retrieved"], model=cross_model,
                                            client=client)
        return item["id"], scores

    with ThreadPoolExecutor(max_workers=6) as ex:
        judged = dict(ex.map(judge_all, built))

    dims = ["groundedness", "relevance", "correctness", "tone", "actionability", "overall"]
    summary = {}
    for sys_name in ["trivial", "simple", "agent", "agent_cross"]:
        summary[sys_name] = {d: round(float(np.mean(
            [judged[i][sys_name][d] for i in judged])), 3) for d in dims}
    for sys_name in ["trivial", "simple", "agent"]:
        s = summary[sys_name]
        print(f"  {sys_name:14} overall={s['overall']:.2f}  ground={s['groundedness']:.2f} "
              f"rel={s['relevance']:.2f} corr={s['correctness']:.2f} "
              f"tone={s['tone']:.2f} act={s['actionability']:.2f}")
    print(f"  {'agent (cross-judge '+cross_model.split('-')[-1]+')':14} "
          f"overall={summary['agent_cross']['overall']:.2f}  "
          f"(primary judge {judge_model.split('-')[-1]}: "
          f"{summary['agent']['overall']:.2f})")

    # Save per-item judgments and the built replies for inspection / human rating.
    with open(OUT / "reply_judgments.jsonl", "w") as f:
        for item in built:
            f.write(json.dumps({**item, "scores": judged[item["id"]]},
                               ensure_ascii=False) + "\n")

    # Judge-human agreement, if human ratings are present.
    human_path = ROOT / "golden" / "reply_human_ratings.jsonl"
    if human_path.exists():
        human = {json.loads(l)["id"]: json.loads(l)["overall"] for l in open(human_path)}
        ids = [i for i in judged if i in human]
        if ids:
            h = [human[i] for i in ids]
            j = [judged[i]["agent"]["overall"] for i in ids]
            agr = M.numeric_agreement(h, j)
            summary["judge_human_agreement"] = agr
            print(f"\n  Judge-human agreement on agent replies (n={agr['n']}): "
                  f"QWK={agr['quadratic_weighted_kappa']} "
                  f"acceptable-kappa={agr['acceptable_kappa']} MAE={agr['mean_abs_error']}")
    else:
        print("\n  (No human reply ratings yet -> run `python -m src.rate` for "
              "judge-human agreement.)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reply-n", type=int, default=60)
    ap.add_argument("--classify-only", action="store_true")
    ap.add_argument("--compare-models", nargs="*", default=None)
    args = ap.parse_args()

    client = gemini_client()
    rows = load_golden()
    results = {}

    if args.compare_models:
        comp = {}
        for mdl in args.compare_models:
            res, _ = eval_classification(rows, client, model=mdl)
            comp[mdl] = res["llm"]
        results["model_comparison"] = comp
        print("\n=== CLASSIFIER MODEL COMPARISON ===")
        for mdl, s in comp.items():
            print(f"  {mdl:26} acc={s['accuracy']:.3f}  macroF1={s['macro_f1']:.3f}")
    else:
        results["classification"], llm_pred = eval_classification(rows, client)
        if not args.classify_only:
            results["escalation"], esc_pred = eval_escalation(rows, client)
            results["replies"] = eval_replies(rows, client, args.reply_n)
            # Per-row dump for failure analysis.
            with open(OUT / "per_row.jsonl", "w") as f:
                for r, pi, pd_ in zip(rows, llm_pred, esc_pred):
                    f.write(json.dumps({
                        "id": r["cust_tweet_id"], "message": r["customer_text"],
                        "gold_intent": r["intent"], "pred_intent": pi,
                        "gold_decision": r["decision"], "pred_decision": pd_,
                        "difficulty": r.get("difficulty"),
                        "reference_reply": r["reference_reply"],
                    }, ensure_ascii=False) + "\n")

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
