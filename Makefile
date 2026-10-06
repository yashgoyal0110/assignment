.PHONY: install data taxonomy golden embed eval demo clean

install:
	python3.12 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

# Full rebuild from the raw Kaggle file (needs Kaggle creds). Optional:
# the subsample under data/sample is already committed.
data:
	.venv/bin/python -m src.download
	.venv/bin/python -m src.prepare build --brand AmazonHelp

# Pre-label a stratified sample, then review by hand.
golden:
	.venv/bin/python -m src.label propose
	@echo "Now run: .venv/bin/python -m src.label review"

# Headline results: classification + escalation + reply quality.
eval:
	.venv/bin/python -m eval.run_eval --reply-n 30

demo:
	.venv/bin/python scripts/demo.py

clean:
	rm -rf outputs data/cache __pycache__ src/__pycache__ eval/__pycache__