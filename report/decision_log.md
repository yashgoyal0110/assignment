# Decision log

Non-obvious choices and why. In rough order of how much they shaped the result.

1. **Brand = AmazonHelp.** Highest support volume in the dataset (~170k replies),
   so the retrieval grounding corpus is the richest. Intents are broad but cleanly
   separable, and there is a natural auto-vs-escalate split. Deliberately avoided
   AppleSupport: its public replies are overwhelmingly "DM us so we can help,"
   which makes grounded drafting repetitive and the reply-quality eval weak.

2. **Scoped to English only.** Amazon's data is heavily multilingual (French,
   German, Spanish, Portuguese). One brand, one taxonomy, one judge is honest and
   defensible; pretending a single English taxonomy handles all languages is not.
   Language filtering is done with `langdetect` (seeded for determinism).

3. **Scoped to thread-initial (first-contact) messages.** Many tweets are
   mid-thread fragments ("thanks, will try later", "still not solved") with no
   standalone intent. Classifying those is guesswork. We keep only messages that
   open a thread, and treat a context-less follow-up as an escalation signal
   rather than something to classify.

4. **Staged pipeline, not an autonomous multi-agent system.** The task is a fixed
   three-stage flow (classify, ground, decide). Specialized prompted calls keep
   every stage independently measurable and the decision auditable. Autonomous
   agents handing off to each other would make the metrics and the failure
   analysis much harder to pin down.

5. **Held the eval pool out of the grounding corpus.** If an eval message's own
   historical reply were in the retrieval index, the drafter could just copy it
   and the reply-quality score would be inflated. The split is done at data-prep
   time (`src/prepare.py`), 80% grounding / 20% eval pool.

6. **Escalation gate is rule-based, not another LLM call.** Reasons are then
   transparent and deterministic, which matters for a trust decision and for the
   live walkthrough. An LLM gate is listed as future work.

7. **Did not use the classifier's self-reported confidence to drive escalation.**
   It saturates at 1.00 and is not calibrated. The real uncertainty signal is
   top retrieval similarity: a message unlike anything the brand has handled has
   no precedent, so it escalates.

8. **`fraud_security` is a hard always-escalate class.** Phishing, scams, malware,
   and account-security concerns go to a human no matter how confident the model
   is. Keeping it as its own intent makes that rule clean.

9. **Golden labels are model-proposed but human-verified.** The strong `pro` model
   proposes labels to save time, but a label only counts as gold after a human
   confirms it. Using the flash classifier's own outputs as its answer key would
   be circular; proposing with a different model (pro) reduces baked-in agreement,
   and human verification removes the rest. We report how often the human had to
   correct the proposal.

10. **Eight intents, kept small on purpose.** More classes would thin per-class
    support in a ~200-example golden set and make macro-F1 noisy. Eight covers the
    data (checked by reading ~65 real messages) with a `general_other` catch-all.

11. **Cross-model judge.** Reply quality is scored by `pro` and re-scored by a
    cheaper `flash-lite` judge. The gap between the two is reported, so the
    headline quality number is not just one model grading its own family.

12. **Judge is validated against a human.** An LLM judge is only worth trusting if
    it agrees with a person. We measure agreement (quadratic-weighted and
    binarized Cohen's kappa) on a human-rated subset of agent replies.

13. **Baselines bracket the problem.** Trivial (majority class / canned reply /
    always-auto) and simple (TF-IDF + logistic regression / nearest-neighbour copy
    of a real past reply). The nearest-neighbour copy is a deliberately strong
    reply baseline: it returns an actual human reply, so beating it is meaningful.

14. **Classification compared by stratified 5-fold CV.** The supervised baselines
    are evaluated out-of-fold so they are never scored on their own training data.
    The LLM classifier is zero-shot, so it sees no labels at all, which makes a
    zero-shot win a strong result.

15. **Models are config-driven.** Every stage's model is a line in `config.yaml`,
    so the eval harness can re-run the whole pipeline under a different Gemini tier
    and report metrics per model (`--compare-models`).

16. **Embedding cache keyed on corpus content + model.** Reruns are instant and a
    stale cache can never silently load against a changed corpus.
