"""The support agent: classify -> retrieve -> decide -> draft.

Even when the gate says 'escalate', we still draft a suggested reply so a human
has a starting point. The pipeline returns one record with every stage's output,
which is what the eval harness and the demo consume.
"""
from src.classify import classify
from src.retrieve import retrieve
from src.escalate import decide
from src.draft import draft
from src.common import gemini_client, config


class Agent:
    def __init__(self, models: dict = None):
        """models can override config per stage, so the eval harness can run the
        whole pipeline under a different Gemini tier."""
        self.client = gemini_client()
        self.models = models or {}

    def handle(self, message: str) -> dict:
        m = {**config()["models"], **self.models}

        intent = classify(message, client=self.client, model=m["classify"])
        retrieved = retrieve(message, client=self.client, model=m["embed"])
        decision = decide(message, intent, retrieved)
        reply = draft(message, intent["intent"], retrieved,
                      client=self.client, model=m["draft"])

        return {
            "message": message,
            "intent": intent["intent"],
            "confidence": intent["confidence"],
            "intent_rationale": intent["rationale"],
            "decision": decision["decision"],
            "decision_reason": decision["reason"],
            "decision_rule": decision["rule"],
            "max_similarity": decision["max_similarity"],
            "reply": reply,
            "retrieved": [
                {"customer_text": r["customer_text"], "brand_reply": r["brand_reply"],
                 "similarity": round(r["similarity"], 3)} for r in retrieved
            ],
        }


if __name__ == "__main__":
    import sys, json
    a = Agent()
    msg = sys.argv[1] if len(sys.argv) > 1 else "my parcel never arrived, paid for next day"
    rec = a.handle(msg)
    for k in ["message", "intent", "decision", "decision_reason", "reply"]:
        print(f"{k:16}: {rec[k]}")
