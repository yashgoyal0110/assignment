"""Intent taxonomy for AmazonHelp, derived by reading ~65 real customer
messages from the subsample (see report for the sampling note).

Design choices:
- 8 labels. Kept small on purpose: with a 150-250 example golden set, more
  classes means too few examples per class for a trustworthy macro-F1.
- `fraud_security` is separated out because it must always escalate, no matter
  how confident the classifier is.
- `general_other` is the honest catch-all: praise, venting with no actionable
  request, and general policy/info questions that don't map to an operation.

Each intent carries a default disposition used by the escalation gate as a
prior. The gate still overrides it using confidence, grounding, and signals.
"""

INTENTS = {
    "delivery_issue": {
        "desc": "Late, missing, failed, or damaged-in-transit delivery; tracking "
                "problems; packaging complaints.",
        "examples": [
            "Paid for next day delivery, but it didn't arrive. Need it for Monday.",
            "My subscribe & save order has not been delivered, apparently damaged in transit.",
            "I have ordered something for £3, hasn't arrived, ordered it on 1 August.",
        ],
        "default": "auto",   # can give tracking/next-steps guidance
    },
    "return_refund": {
        "desc": "Returns, replacements for defective/wrong items, and refund status "
                "or disputes (getting goods or money back after purchase).",
        "examples": [
            "It's been 6 days and no one is coming to collect my return.",
            "I received a defective set, now I want a replacement.",
            "Order cancelled on 4th. Called 4 times, yet to get the refund.",
        ],
        "default": "escalate",   # usually needs account/order lookup
    },
    "order_management": {
        "desc": "Placing, cancelling, modifying, or checking the status of an order; "
                "pre-purchase questions about an order.",
        "examples": [
            "Should I cancel and await refund and re-order, or will it arrive tomorrow?",
            "Why can't I cancel this order that won't ship in any reasonable time?",
            "If I order this today on your app, can it come by tomorrow?",
        ],
        "default": "auto",
    },
    "billing_payment": {
        "desc": "Unexpected charges, pricing transparency, payment methods, EMI, and "
                "costs (money going out, as opposed to refunds coming back).",
        "examples": [
            "Why do you give final prices, handling, taxes only after entering CVV and OTP?",
            "No follow from amazon on no cost emi. Blaming bank.",
            "I was charged for 4 items but I only added 3 to my cart.",
        ],
        "default": "escalate",
    },
    "account_access": {
        "desc": "Login problems, account closed/deleted/locked, account or profile "
                "settings, seller accounts, reviews, and Prime/membership management.",
        "examples": [
            "A few weeks back you deleted my seller account accidentally, no replies since.",
            "Every review I have ever left has been removed. Can you tell me why?",
            "Is this how you get people to pay for prime? By not letting them cancel the trial?",
        ],
        "default": "escalate",   # account-specific, often sensitive
    },
    "product_device_tech": {
        "desc": "Amazon device, app, or digital content not working; technical "
                "troubleshooting (Kindle, Fire, Prime Video/Music playback, app bugs).",
        "examples": [
            "Every time I look at a kindle book it offers a sample but won't let me BUY it.",
            "Video quality is horrible for the game.",
            "Apps don't load. Main screen is blank with a spinning wheel.",
        ],
        "default": "auto",   # troubleshooting steps are safe to send
    },
    "fraud_security": {
        "desc": "Phishing, scams, fraudulent sellers, malware, or account-security "
                "concerns. Always routed to a human.",
        "examples": [
            "Hey, got a phishing mail.",
            "Your seller will refund only if I give a good review. Is this Amazon policy?",
            "My dad bought a phone from you, it's infected with malware that installs apps.",
        ],
        "default": "escalate",   # hard rule in the gate
    },
    "general_other": {
        "desc": "Praise, thanks, venting with no actionable request, and general "
                "policy/info questions that don't map to a specific operation.",
        "examples": [
            "Got to hand it to you for getting my package from the US to the UK in 3 days!",
            "When did prime change from 2 day to 1 week?",
            "Can everybody in the family use Amazon music coming with my prime account?",
        ],
        "default": "auto",
    },
}

LABELS = list(INTENTS.keys())
ALWAYS_ESCALATE = {"fraud_security"}


def taxonomy_prompt_block() -> str:
    """Render the taxonomy as a block for classifier/labeling prompts."""
    lines = []
    for name, spec in INTENTS.items():
        lines.append(f"- {name}: {spec['desc']}")
        for ex in spec["examples"]:
            lines.append(f"    e.g. \"{ex}\"")
    return "\n".join(lines)
