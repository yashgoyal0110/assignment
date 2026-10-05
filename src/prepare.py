"""Inspect brand volumes and build the working subsample for one brand.

  python -m src.prepare inspect
  python -m src.prepare build --brand AmazonHelp

'build' reconstructs (customer message -> brand reply) pairs, cleans them,
subsamples, and splits into a grounding corpus and a held-out eval pool.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

from src.common import config, ROOT

DetectorFactory.seed = 0

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\s+")


def find_csv() -> str:
    hits = glob.glob(str(ROOT / "data" / "raw" / "**" / "twcs.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError("twcs.csv not found under data/raw. Run: python -m src.download")
    return hits[0]


def clean(text: str) -> str:
    text = URL.sub("<url>", str(text))
    text = MENTION.sub("", text)
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return WS.sub(" ", text).strip()


def is_english(text: str) -> bool:
    t = text.replace("<url>", "").strip()
    if len(t.split()) < 3:
        return False
    try:
        return detect(t) == "en"
    except LangDetectException:
        return False


def load(usecols=None) -> pd.DataFrame:
    return pd.read_csv(find_csv(), usecols=usecols, dtype={"tweet_id": str,
                       "in_response_to_tweet_id": str})


def inspect(top=25):
    df = load(usecols=["author_id", "inbound"])
    brands = df[df["inbound"] == False]  # noqa: E712
    counts = brands["author_id"].value_counts().head(top)
    print(f"Top {top} brands by number of outbound (support) tweets:\n")
    for name, n in counts.items():
        print(f"  {n:>8,}  {name}")
    print("\nPick one and run: python -m src.prepare build --brand <name>")


def build(brand: str):
    cfg = config()
    df = load(usecols=["tweet_id", "author_id", "inbound", "created_at", "text",
                       "in_response_to_tweet_id"])
    by_id = df.set_index("tweet_id", drop=False)

    replies = df[(df["author_id"] == brand) & (df["inbound"] == False) &  # noqa: E712
                 (df["in_response_to_tweet_id"].notna())]
    if len(replies) == 0:
        raise SystemExit(f"No outbound replies found for brand '{brand}'. "
                         f"Run 'inspect' to see valid names (case-sensitive).")

    pairs, seen = [], set()
    for _, r in replies.iterrows():
        pid = r["in_response_to_tweet_id"]
        if pid not in by_id.index:
            continue
        parent = by_id.loc[pid]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if not parent["inbound"]:
            continue
        # Thread-initial only: skip mid-conversation follow-ups (no standalone intent).
        if pd.notna(parent["in_response_to_tweet_id"]):
            continue
        if pid in seen:
            continue
        seen.add(pid)
        cust = clean(parent["text"])
        reply = clean(r["text"])
        n_words = len(cust.split())
        if n_words < 3 or n_words > 80 or not reply:
            continue
        if not is_english(cust):
            continue
        pairs.append({
            "cust_tweet_id": pid,
            "created_at": parent["created_at"],
            "customer_text": cust,
            "brand_reply": reply,
        })

    print(f"{brand}: {len(replies):,} replies -> {len(pairs):,} clean "
          f"(customer -> reply) pairs")

    sample_size = min(cfg["data"]["sample_size"], len(pairs))
    pdf = pd.DataFrame(pairs).sample(n=sample_size, random_state=cfg["data"]["random_seed"])
    n_eval = int(sample_size * 0.2)
    eval_pool = pdf.iloc[:n_eval]
    grounding = pdf.iloc[n_eval:]

    out = ROOT / cfg["data"]["sample_dir"]
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "grounding.jsonl", grounding)
    _write_jsonl(out / "eval_pool.jsonl", eval_pool)
    print(f"Wrote {len(grounding):,} grounding + {len(eval_pool):,} eval-pool rows to {out}")

    _set_config_brand(brand)
    print(f"Set brand.name = {brand} in config.yaml")


def _write_jsonl(path: Path, frame: pd.DataFrame):
    with open(path, "w") as f:
        for _, row in frame.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def _set_config_brand(brand: str):
    path = ROOT / "config.yaml"
    text = path.read_text()
    text = re.sub(r"(?m)^(\s*name:).*$", rf"\1 {brand}", text, count=1)
    path.write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    b = sub.add_parser("build")
    b.add_argument("--brand", required=True)
    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect()
    else:
        build(args.brand)
