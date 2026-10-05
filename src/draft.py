"""Stage 3: draft a reply grounded in the brand's historical resolutions."""
from src.common import config, gemini_client, generate_text

PROMPT = """You are a customer support agent replying on Twitter as {brand}.
Write a single reply to the customer's message below.

Ground your reply in how {brand} has actually handled similar cases. Here are
the most similar past cases and the replies that were sent:

{examples}

Guidelines:
- Match the tone of the past replies: brief, warm, specific. One short paragraph.
- Do NOT invent order numbers, dates, refund amounts, or account details.
- If resolving needs account-specific info, ask for it or point to the standard
  next step, exactly as the past replies did.
- Do not promise outcomes the past replies did not promise.
- Plain text only, no hashtags, no signature.

Detected intent: {intent}

Customer message:
\"\"\"{message}\"\"\"

Reply:"""


def draft(message: str, intent: str, retrieved: list, client=None, model=None) -> str:
    cfg = config()
    client = client or gemini_client()
    model = model or cfg["models"]["draft"]
    examples = "\n".join(
        f'- Customer: "{r["customer_text"][:200]}"\n  Reply: "{r["brand_reply"][:200]}"'
        for r in retrieved
    )
    prompt = PROMPT.format(brand=cfg["brand"]["name"], examples=examples,
                           intent=intent, message=message)
    return generate_text(client, model, prompt, temperature=0.3)


if __name__ == "__main__":
    from src.retrieve import retrieve
    msg = "my package never arrived and I paid for prime next day delivery"
    r = retrieve(msg)
    print(draft(msg, "delivery_issue", r))
