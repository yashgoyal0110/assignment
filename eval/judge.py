"""LLM-as-judge for reply quality. Scores a reply on five dimensions plus an
overall, given the message and the brand's similar past cases as grounding.
Run with two judge models (pro + a flash-lite cross-judge) to check bias."""
from pydantic import BaseModel

from src.common import config, gemini_client, generate_json


class Rubric(BaseModel):
    groundedness: int    # consistent with how the brand actually handles this
    relevance: int       # addresses the customer's actual issue
    correctness: int     # no invented orders/policies/promises
    tone: int
    actionability: int
    overall: int         # would a support lead send this as-is
    justification: str


PROMPT = """You are a strict QA reviewer for {brand}'s customer support replies.
Rate the DRAFT REPLY to the customer message on each dimension from 1 (poor) to
5 (excellent). Judge groundedness against how the brand actually handled similar
cases, shown below. Penalise invented order numbers, policies, or promises hard.

Similar past cases (the brand's real handling):
{examples}

Customer message:
\"\"\"{message}\"\"\"

Draft reply to rate:
\"\"\"{reply}\"\"\"

Score every dimension 1-5. overall is your holistic verdict: would a support
lead send this as-is (5) or not at all (1). Give a one-sentence justification."""


def judge_reply(message, reply, retrieved, model=None, client=None) -> dict:
    cfg = config()
    client = client or gemini_client()
    model = model or cfg["models"]["judge"]
    examples = "\n".join(
        f'- Customer: "{r["customer_text"][:160]}" -> Reply: "{r["brand_reply"][:160]}"'
        for r in retrieved
    )
    prompt = PROMPT.format(brand=cfg["brand"]["name"], examples=examples,
                           message=message, reply=reply)
    out = generate_json(client, model, prompt, Rubric)
    for k in ["groundedness", "relevance", "correctness", "tone",
              "actionability", "overall"]:
        out[k] = max(1, min(5, int(out.get(k, 3))))
    return out
