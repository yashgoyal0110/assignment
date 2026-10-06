# AmazonHelp support agent: report

## 1. Problem framing

The brand is AmazonHelp. The agent handles one incoming customer tweet at a time
and does three things: classify the intent, draft a reply grounded in how Amazon
has resolved similar issues, and decide whether to auto-handle or escalate, with
a reason.

**What "good" means here.** For a support brand on Twitter, the expensive
mistakes are not slightly-off wordings, they are: auto-replying to something that
needed a human (a refund dispute, a fraud report, an angry customer threatening
legal action), and hallucinating specifics (an order status, a refund amount, a
policy) that Amazon never promised. So "good" is weighted toward:
- high escalation recall on the things that must reach a human (fraud, money,
  legal, account-specific action), even at the cost of escalating some easy ones,
- replies that are grounded and make no promises the brand would not make,
- correct intents, because the intent gates everything downstream.

A fast, fluent reply that quietly invents an order number is worse than a dull
reply that routes correctly. The metrics below reflect that.

**What I chose not to build.**
- No multilingual support. Amazon's data is heavily multilingual; I scoped to
  English so one taxonomy and one judge stay meaningful.
- No multi-turn conversation state. The agent handles first-contact messages;
  mid-thread follow-ups are out of scope and are themselves an escalation signal.
- No live account lookups or actions. The agent drafts and routes; it never
  pretends to have order data.
- No autonomous multi-agent system. A staged pipeline is more measurable and
  more defensible for a trust decision.
- No fine-tuning. Zero-shot classification and retrieval-grounded drafting.

## 2. Results vs baselines

Golden set: 220 examples. Classification uses stratified 5-fold cross-validation
so the supervised baseline is never scored on its training data; the LLM
classifier is zero-shot. Labels are model-made and cross-checked (see section 4).

### Intent classification (macro-F1 over 8 intents)

| system | accuracy | macro-F1 |
|---|---|---|
| trivial (always majority class) | 0.396 | 0.071 |
| simple (TF-IDF + logistic regression) | 0.441 | 0.312 |
| agent (Gemini flash, zero-shot) | **0.850** | **0.861** |

Per-class the agent is strong on the specific intents (fraud_security 1.00,
delivery_issue 0.91, return_refund 0.90) but weak on the catch-all:
`general_other` precision is only 0.55 (see failure 1).

### Escalation decision (positive class = escalate)

| system | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| trivial (always auto) | 0.295 | 0.000 | 0.000 | 0.000 |
| agent (rule gate) | 0.514 | **0.833** | 0.387 | 0.529 |

This is the least flattering and most informative result. When the gate escalates
it is usually right (precision 0.83), but it catches only 39% of what the gold
set says should escalate. The agent under-escalates (see failure 2). Note the
gold set is escalate-heavy (155/220), so some of this gap is a strict standard,
not a broken gate (discussed in section 4).

### Reply quality (LLM-judge, 1-5, n=30)

| system | overall | grounded | relevant | correct | tone | actionable |
|---|---|---|---|---|---|---|
| trivial (canned reply) | 2.07 | 2.37 | 3.27 | 2.93 | 3.40 | 3.03 |
| simple (copy nearest past reply) | 3.13 | 3.70 | 3.30 | 3.27 | 3.77 | 3.47 |
| agent (grounded draft) | **5.00** | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |

Primary judge: `gemini-pro-latest`. Cross-judge (`gemini-3.5-flash-lite`) on the
agent replies: overall 4.97 vs primary 5.00. The agent clearly beats both
baselines, but a flat 5.00 on every dimension is a judge problem, not proof of
perfection (see failure 5 and section 4).

### Label reliability and judge agreement
- Inter-annotator agreement on the golden labels (pro vs flash): intent
  kappa **0.885** (exact 0.909), decision kappa **0.770** (exact 0.905). Both are
  in the "substantial" range, so the labels are stable; the decision kappa being
  lower confirms auto-vs-escalate is the genuinely harder judgment.
- Judge-human agreement: not yet measured. With no human reply ratings, the only
  agreement available is judge-vs-judge (pro vs flash-lite), which shows
  consistency, not correctness. A 10-minute human rating pass (`python -m
  src.rate`) produces the real Cohen's kappa; it is the top next step.

## 3. Failure analysis (top 5)

**1. The classifier over-uses `general_other`.** 20 of 33 intent errors are an
operational intent collapsing into the catch-all: 13x delivery_issue, 4x
order_management, 3x return_refund. Example: *"So, I guess this is how
#AmazonPrime delivers now, boxes randomly tossed into customers' yards?"* gold
delivery_issue, predicted general_other. Hypothesis: the flash model keys on the
absence of an explicit request (it is venting, not asking) and routes to the
catch-all, instead of the underlying topic. This drags general_other precision to
0.55 and is the single biggest classification weakness.

**2. The gate under-escalates angry complaints (recall 0.39).** 95 messages the
gold marks escalate were auto-handled. They cluster on emotional delivery_issue
venting with no fraud/legal keyword and good retrieval grounding, so every gate
rule passes them through. Example: *"fed up pls look ... I want my products dont
need apologies. more than 5 apologies"* gold escalate, gate auto. Hypothesis: the
gate has no anger or repeat-contact signal, and delivery_issue defaults to auto,
so a furious repeat complaint looks identical to a calm first question.

**3. The gate over-escalates simple questions in escalate-default intents.** 12
false alarms, all one-line informational questions. Example: *"are returns
through Royal Mail free?"* gold auto, gate escalate (because return_refund
defaults to escalate). Hypothesis: the intent-prior rule is too blunt. It
escalates the whole class rather than distinguishing a policy question from an
account-specific dispute.

**4. The gate's decision contradicts the drafted reply.** For many delivery
complaints the gate says auto, yet the drafted reply itself routes the customer
to a human channel ("please reach out to us via phone or chat here"). So the
"auto" reply is not a self-contained resolution, it is a hand-off. Hypothesis:
the drafter faithfully imitates Amazon's real replies, which route almost
everything to DM, so even auto cases produce escalation-shaped text. The gate and
the drafter are using different implicit definitions of "handled."

**5. The judge cannot tell good from excellent.** The agent scores 5.00 on every
dimension for all 30 replies, including borderline cases. Example: to *"why am I
paying for Prime if two-day shipping takes more than two days"* the agent asserts
*"Prime Two-Day Shipping refers to transit time once an item ships"*, a policy
claim that could be subtly wrong, and correctness still scores 5. Hypothesis: the
rubric has no negative anchors and the judge shares a model family with the
drafter, so it is lenient and uncalibrated at the top of the scale.

## 4. What is misleading about my headline number?

The most flattering number is the agent's intent macro-F1 of **0.861**. Several
things inflate it:

1. **The golden labels are model-made, not human.** They were produced by a
   strong model (pro) and the classifier under test is a cheaper model from the
   same family (flash). Part of the agreement is two Gemini models agreeing, not
   the agent being correct. Inter-annotator kappa of 0.885 bounds how much the
   F1 can mean; a human set would likely pull it down, most on the
   general_other boundary.
2. **The reply-quality judge saturates at 5.00.** A score that cannot go down is
   barely measuring. The cross-judge agreeing (4.97) shows consistency between
   two Gemini models, not calibration to a human.
3. **Class imbalance hides behind the averages.** delivery_issue is 87 of 220.
   The rare high-stakes classes (fraud_security n=7, product_device_tech n=14)
   have wide error bars that a single macro-F1 smooths over.
4. **The escalation story cuts both ways.** The honest F1 is 0.529 (the earlier
   0.966 was the gate grading its own labels, which I removed). But the gold is
   escalate-heavy because the labelling rubric treats Amazon's DM-routing as an
   escalate signal, and Amazon DMs almost everything. So the gate's low recall is
   partly a genuinely too-passive gate (failure 2) and partly a gold standard
   that over-escalates. The true number depends on a definition reasonable people
   would argue about, which is itself the point.
5. **One brand, one subsample, one language.** Nothing here shows the agent
   generalises across brands, time, or the multilingual tail I removed.

Honest one-liner: the agent is clearly far better than both baselines on every
axis, but the exact figures are upper bounds set by model-made labels and a
lenient judge, not human-validated truth. The escalation recall is the number I
trust most as a real weakness.

## 5. What I would do next with one more week

1. **A real human-labelled golden set.** Hand-label 300 examples (or two
   annotators plus adjudication) and re-run everything. This single change makes
   every number trustworthy. The tooling exists (`src.label review`, `src.rate`).
2. **Fix the escalation gate.** Add an anger/sentiment and repeat-contact signal
   so furious complaints escalate (failure 2), and split escalate-default intents
   so simple policy questions stay auto (failure 3). Reconcile the gate with the
   drafter so an "auto" reply actually resolves rather than routes (failure 4).
3. **Calibrate the judge.** Add deliberately bad replies as negative anchors,
   measure judge-human kappa, and swap in a non-Gemini judge to remove family
   bias and the 5.00 saturation.
4. **Real confidence.** Replace the saturated self-reported confidence with
   self-consistency sampling, giving the gate a genuine uncertainty signal.
5. **Multi-turn and multilingual.** Handle follow-ups with thread context and add
   one more language end to end to check the design holds.
