# AmazonHelp support agent

An AI customer-support agent for one brand (AmazonHelp) built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. For each incoming customer message it:

1. classifies the message into one of 8 intents,
2. drafts a reply grounded in how Amazon has actually resolved similar issues,
3. decides auto-handle vs escalate-to-human, with a stated reason.

The point of the repo is not the agent, it is the evidence that the agent works.
See [report/report.md](report/report.md) for results, baselines, failure analysis,
and what the headline number hides. See [report/decision_log.md](report/decision_log.md)
for the non-obvious choices.

## Architecture

A staged pipeline, one Gemini tier per stage, not an autonomous multi-agent
system. Each stage is a separate, independently measurable call.

```
message
  -> classify   (gemini-3.5-flash, structured output)        src/classify.py
  -> retrieve   (gemini-embedding-001, cosine over history)  src/retrieve.py
  -> decide     (rule-based gate, stated reason)             src/escalate.py
  -> draft      (gemini-3.8-flash, grounded in retrieved)    src/draft.py
```

Reply quality is judged by a different, stronger tier (`gemini-pro-latest`) than
the one that writes the replies, and cross-checked by a cheap third model, so the
quality score is not one model grading its own family. Every model is a line in
[config.yaml](config.yaml).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then put your GEMINI_API_KEY in .env
```

A Gemini API key is the only hard requirement to reproduce results. Kaggle
credentials are needed only if you want to rebuild the subsample from the raw
3M-row file; the subsample under `data/sample/` is already committed.

## Reproduce the headline results (under 15 minutes)

The hand-labelled golden set (`golden/golden_set.jsonl`) and the brand subsample
are committed, so you do not need the full Kaggle download.

```bash
make eval        # classification + escalation + reply quality
# or:
.venv/bin/python -m eval.run_eval --reply-n 30
```

First run embeds the 3,200-message grounding corpus once (~2.5 min) and caches it.
Results print to the console and are written to `outputs/metrics.json`.

Try the agent on your own messages:

```bash
make demo
.venv/bin/python scripts/demo.py "my parcel never arrived and I paid for next day"
```

## Rebuild everything from scratch (optional)

```bash
.venv/bin/python -m src.download                      # needs Kaggle creds
.venv/bin/python -m src.prepare build --brand AmazonHelp
.venv/bin/python -m src.label propose                 # model-proposed labels
.venv/bin/python -m src.label review                  # human verification (the gold step)
.venv/bin/python -m eval.run_eval
.venv/bin/python -m src.rate                           # human reply ratings -> judge agreement
```

## Compare models

```bash
.venv/bin/python -m eval.run_eval --compare-models gemini-3.5-flash gemini-3.5-flash-lite
```

## Layout

```
src/
  prepare.py    brand selection, thread reconstruction, English + first-contact filter
  taxonomy.py   the 8 intents, derived from reading real messages
  classify.py   intent classifier
  retrieve.py   embedding index + cosine retrieval (cached)
  escalate.py   rule-based auto/escalate gate
  draft.py      grounded reply drafter
  agent.py      the pipeline, end to end
  baselines.py  trivial + simple baselines (classification and reply)
  label.py      build + human-verify the golden set
  rate.py       human reply ratings for judge agreement
eval/
  run_eval.py   the harness (classification, escalation, reply quality)
  judge.py      LLM-as-judge rubric
  metrics.py    accuracy, macro-F1, escalation P/R/F1, Cohen's kappa
golden/         the hand-labelled evaluation set + labelling note
report/         report and decision log
```

## Credits / borrowed

- Dataset: thoughtvector/customer-support-on-twitter (Kaggle).
- Gemini via the `google-genai` SDK; scikit-learn for baselines and metrics;
  `langdetect` for language filtering.
