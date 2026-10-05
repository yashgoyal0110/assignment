"""Baselines to compare the agent against.

Classification:
  trivial = always predict the majority intent.
  simple  = TF-IDF + logistic regression (classic supervised text classifier).

Reply:
  trivial = one canned response for everything.
  simple  = copy the single most similar historical brand reply (IR, no LLM).

The LLM classifier is zero-shot (sees no training labels), while the simple
baseline is trained on labels, so a zero-shot win is a meaningful result.
"""
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CANNED_REPLY = ("Thanks for reaching out. Please send us a DM with your order "
                "details and we'll be happy to help.")


def majority_predict(y_train, X_test):
    mode = Counter(y_train).most_common(1)[0][0]
    return [mode] * len(X_test)


def simple_classifier():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def trivial_reply(_message):
    return CANNED_REPLY


def simple_reply(_message, retrieved):
    """Copy the most similar historical reply verbatim."""
    return retrieved[0]["brand_reply"] if retrieved else CANNED_REPLY
