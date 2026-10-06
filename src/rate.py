"""Human rating of agent replies, for judge-human agreement (Cohen's kappa).

Run `python -m eval.run_eval` first (it writes outputs/reply_judgments.jsonl),
then `python -m src.rate` to score the agent replies yourself. The eval harness
picks these up and reports how well the LLM judge agrees with you.
"""
import json
from pathlib import Path

from src.common import ROOT

JUDGMENTS = ROOT / "outputs" / "reply_judgments.jsonl"
OUT = ROOT / "golden" / "reply_human_ratings.jsonl"


def main():
    if not JUDGMENTS.exists():
        raise SystemExit("Run `python -m eval.run_eval` first.")
    items = [json.loads(l) for l in open(JUDGMENTS)]
    done = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            done[r["id"]] = r

    print("\nRate each AGENT reply 1-5 on whether a support lead would send it "
          "as-is.\n5=send as-is, 1=never. Enter to skip, q to save+quit.\n")
    for item in items:
        if item["id"] in done:
            continue
        print("-" * 70)
        print("MSG  :", item["message"])
        print("REPLY:", item["replies"]["agent"])
        ans = input("overall 1-5 / q: ").strip().lower()
        if ans == "q":
            break
        if ans in {"1", "2", "3", "4", "5"}:
            done[item["id"]] = {"id": item["id"], "overall": int(ans)}

    with open(OUT, "w") as f:
        for r in done.values():
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(done)} ratings to {OUT}")


if __name__ == "__main__":
    main()
