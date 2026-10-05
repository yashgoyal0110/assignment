"""Stage 1: intent classification with Gemini Flash + structured output."""
from pydantic import BaseModel

from src.common import config, gemini_client, generate_json
from src.taxonomy import LABELS, taxonomy_prompt_block


class IntentResult(BaseModel):
    intent: str
    confidence: float   # self-reported, not calibrated; not used to drive escalation
    rationale: str


PROMPT = """You are an intent classifier for Amazon's customer support on Twitter.
Classify the customer's message into exactly one of these intents:

{taxonomy}

Rules:
- Pick the single best-fitting intent label from the list above.
- confidence is your probability that the label is correct, 0.0 to 1.0.
- rationale is one short sentence.
- If the message is praise, venting, or a general question with no clear
  operation, use general_other.
- If it mentions phishing, scams, fraud, malware, or account security, use
  fraud_security.

Customer message:
\"\"\"{text}\"\"\"
"""


def classify(text: str, client=None, model=None) -> dict:
    cfg = config()
    client = client or gemini_client()
    model = model or cfg["models"]["classify"]
    prompt = PROMPT.format(taxonomy=taxonomy_prompt_block(), text=text)
    out = generate_json(client, model, prompt, IntentResult)
    if out.get("intent") not in LABELS:
        out["intent"] = "general_other"
        out["confidence"] = min(float(out.get("confidence", 0.3)), 0.4)
    out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    return out


if __name__ == "__main__":
    import sys
    print(classify(sys.argv[1] if len(sys.argv) > 1 else "my order never arrived"))
