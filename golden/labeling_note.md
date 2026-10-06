# Golden set: how it was sampled and labelled

## What it is
`golden_set.jsonl`, 220 AmazonHelp customer messages, each with a hand-checkable
intent label, an auto/escalate decision, a difficulty tag, and the reply Amazon
actually sent (kept as evidence, not used as a target).

## Sampling
- Source: the 20% eval-pool split that was held out of the retrieval grounding
  corpus (see `src/prepare.py`), so no golden message can retrieve its own answer.
- All messages are English, first-contact (thread-opening) tweets to AmazonHelp.
- Stratified: we ran a strong model over 300 pool messages to get a rough intent
  per message, then sampled toward 220 with a floor per intent so the rare classes
  (fraud_security, billing_payment) are represented, not drowned out by the
  dominant delivery_issue class. The final distribution still reflects the natural
  Amazon mix (delivery-heavy).

## Labelling
- Each message was labelled by the strong `pro` model with a rubric that reads
  the message and Amazon's actual reply, then assigns intent, decision, and
  difficulty. The decision rubric is grounded in how Amazon really handled the
  case (routing to DM/phone/form/specialist is treated as a human-needed signal),
  deliberately independent of the agent's own rule gate so the escalation metric
  is not graded against itself.
- Labelling was AI-assisted, which the assignment permits. This is the honest
  caveat: these are strong silver labels, not human gold. To quantify their
  reliability we had a second, different model (`flash`) re-label every example
  and report inter-annotator agreement (Cohen's kappa) in
  `outputs/inter_annotator.json`. High agreement means the labels are stable;
  disagreements flag the genuinely ambiguous cases.
- A human verification pass is wired up (`python -m src.label review`) and is the
  first recommended step before relying on these numbers in production.

## Fields
`cust_tweet_id, customer_text, reference_reply, intent, decision, difficulty,
label_rationale, annotator, verified`.
