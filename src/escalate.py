"""Stage 4: decide auto-handle vs escalate, with a stated reason.

Deliberately rule-based rather than another LLM call. The reasons are then
transparent and defensible, and the decision is deterministic. Each rule
returns the reason it fired on. The golden set's hand-labelled auto/escalate
decisions are the ground truth we measure this against.

Definitions used:
  auto     = the agent can post a complete, helpful public reply (answer,
             troubleshooting, or standard acknowledge-and-route) with no human.
  escalate = a human must take over: account-specific action, money, security,
             legal/sensitive, high complaint risk, or no precedent to lean on.
"""
import re

from src.taxonomy import INTENTS, ALWAYS_ESCALATE
from src.common import config

# Sensitive signals that override everything. Word-boundary matched.
RISK_PATTERNS = [
    r"\blawyer\b", r"\bsolicitor\b", r"\bsue\b", r"\blawsuit\b", r"\bcourt\b",
    r"\blegal\b", r"\bombudsman\b", r"\btrading standards\b", r"\bgdpr\b",
    r"\bdata breach\b", r"\bjournalist\b", r"\bpress\b", r"\bmedia\b",
    r"\bbbc\b", r"\bceo\b", r"\bbezos\b", r"\bdisabled\b", r"\bdisability\b",
    r"\bwheelchair\b", r"\bdied\b", r"\bdeath\b", r"\binjur", r"\bunsafe\b",
    r"\bhazard\b", r"\bfire\b", r"\bfraud\b", r"\bscam\b", r"\bhacked\b",
    r"\bstolen\b", r"\bunauthori", r"\bcharged twice\b",
]
RISK_RE = re.compile("|".join(RISK_PATTERNS), re.IGNORECASE)


def decide(message: str, intent_result: dict, retrieved: list) -> dict:
    cfg = config()["escalation"]
    intent = intent_result["intent"]
    max_sim = retrieved[0]["similarity"] if retrieved else 0.0

    def out(decision, reason, rule):
        return {"decision": decision, "reason": reason, "rule": rule,
                "max_similarity": round(max_sim, 3)}

    # Rule 1: hard always-escalate intents (fraud/security).
    if intent in ALWAYS_ESCALATE:
        return out("escalate", "Fraud or security concern; must be handled by a "
                   "human agent.", "always_escalate_intent")

    # Rule 2: sensitive/legal signal anywhere in the text.
    m = RISK_RE.search(message)
    if m:
        return out("escalate", f"Message raises a sensitive or legal signal "
                   f"(\"{m.group(0)}\"); routing to a human.", "risk_signal")

    # Rule 3: no precedent. The brand has not handled anything like this, so the
    # draft would be ungrounded guesswork.
    if max_sim < cfg["min_grounding_similarity"]:
        return out("escalate", f"No close precedent in brand history (top "
                   f"similarity {max_sim:.2f}); a human should review.",
                   "no_precedent")

    # Rule 4: intent prior. Some intents inherently need account-specific action
    # that a public reply cannot complete.
    if INTENTS[intent]["default"] == "escalate":
        return out("escalate", f"Intent '{intent}' typically needs account-specific "
                   f"action that a public reply can't complete.", "intent_prior")

    # Otherwise: auto-handle.
    return out("auto", f"Intent '{intent}' with a close precedent (similarity "
               f"{max_sim:.2f}); safe to auto-reply.", "default_auto")
