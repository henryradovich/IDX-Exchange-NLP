#!/usr/bin/env python3
"""Week 7: language-only search-query intent classification.

A word TF-IDF model is augmented with transparent, binary linguistic cues.
The cues are FEATURES for logistic regression, not deterministic classification
rules. Probabilities are model estimates, not evidence of buyer behavior.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

LABELS = ("browsing", "researching", "high_intent_inquiry")

# Designed around the task's *linguistic* definitions: exploratory search,
# information seeking/comparison, and explicit time-sensitive inquiry/action.
# Terms are examples of grammatical signals rather than a behavioral CRM model.
# They are intentionally broad: logistic regression learns their importance.
CUE_PATTERNS = {
    "action_verbs": r"\b(?:schedule|book|arrange|request|reserve|set up)\b",
    "viewing_actions": r"\b(?:tour|showings?|viewings?|open houses?|walkthrough|visits?)\b",
    "offer_actions": r"\b(?:make|submit|place|put in|write|present)\s+(?:an?\s+)?offer\b",
    "contact_verbs": r"\b(?:contact|speak to|talk to|connect(?:\s+me)?\s+with|reach out to|call|message)\b",
    "agent_or_seller": r"\b(?:listing agent|seller|realtor|real estate agent)\b",
    "immediate_timeframe": r"\b(?:this weekend|tomorrow|today|this saturday|this sunday|before next month|this month|asap|immediately|soon)\b",
    "readiness_phrases": r"\b(?:ready\s+to|need\s+(?:a|to)|want\s+to\s+(?:buy|purchase|make|move|close)|move[ -]?in ready|turnkey|ready to close)\b",
    "purchase_verbs": r"\b(?:purchase|buy|close|closing)\b",
    "financing_action": r"\b(?:seller financing|apply for financing|preapproval|pre-approved|prequalified)\b",
    "availability": r"\b(?:available|availability)\b",
    "price_limit": r"\b(?:under|below|up to|within budget|less than)\s+\$?\d",
    "comparison": r"\b(?:compare|comparison|versus|vs\.?|differences?|pros and cons|better than)\b",
    "how_questions": r"\b(?:how much|how many|how have|how do|how has|how does|what is the average|what are the average)\b",
    "what_and_which_questions": r"\b(?:what (?:are|is|should)|which (?:areas?|neighborhoods?|districts?))\b",
    "research_vocabulary": r"\b(?:hoa|tax(?:es)?|fees?|costs?|transit|walkable|walkability|school districts?|flood risks?|resale value|property values?|price per|first time buyers?|market trends?|neighborhoods?|affordability|zoning)\b",
    "property_market_analysis": r"\b(?:average price|median price|price per square foot|property values?|home prices?|resale value|closing costs?)\b",
    "learning_verbs": r"\b(?:research|explain|learn|understand|what should i know|is .* a good place|are there .* risks?)\b",
    "question_opening": r"^(?:what|which|why|how|is|are|does|do)\b",
    "browsing_phrases": r"\b(?:browse|explore|looking around|just exploring|show me|let me see|list houses?|list properties|any nice|show all|what properties|what types of homes|what is on the market|find homes|find houses)\b",
    "listing_nouns": r"\b(?:listings?|properties|homes|houses|condos?|townhomes?|townhouses?)\b",
    "discovery_verbs": r"\b(?:show|find|see|list|look|browse)\b",
    "market_terms": r"\b(?:for sale|on the market|listed|new construction|luxury homes)\b",
}
COMPILED_CUES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in CUE_PATTERNS.values())


class LinguisticCueFeatures(BaseEstimator, TransformerMixin):
    """Extract reproducible sparse cue features, independent of labeled rows."""
    def __init__(self, weight: float = 2.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        items = list(X)
        matrix = np.zeros((len(items), len(COMPILED_CUES)), dtype=np.float32)
        for row, query in enumerate(items):
            query = str(query)
            for col, expression in enumerate(COMPILED_CUES):
                matrix[row, col] = float(bool(expression.search(query)))
        return csr_matrix(matrix * self.weight)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"cue__{name}" for name in CUE_PATTERNS], dtype=object)


class QueryIntentClassifier:
    """TF-IDF plus interpretable linguistic features and logistic regression."""
    def __init__(
        self,
        max_features: int = 5000,
        random_state: int = 42,
        use_cues: bool = True,
        cue_weight: float = 2.0,
        C: float = 1.0,
    ):
        self.max_features = max_features
        self.random_state = random_state
        self.use_cues = use_cues
        self.cue_weight = cue_weight
        self.C = C
        word_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        if use_cues:
            self.vectorizer = FeatureUnion([
                ("word_tfidf", word_vectorizer),
                ("linguistic_cues", LinguisticCueFeatures(weight=cue_weight)),
            ])
        else:
            self.vectorizer = word_vectorizer
        self.model = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=C, random_state=random_state
        )
        self.labels = list(LABELS)
        self.is_fitted = False

    def train(self, queries: Sequence[str], labels: Sequence[str]):
        queries, labels = list(queries), list(labels)
        if len(queries) != len(labels) or not queries:
            raise ValueError("Provide nonempty queries and labels of matching length")
        if any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("All queries must be nonempty strings")
        invalid = set(labels) - set(self.labels)
        if invalid:
            raise ValueError(f"Unknown intent labels: {sorted(invalid)}")
        if set(labels) != set(LABELS):
            raise ValueError("Training must include all three intent classes")
        X = self.vectorizer.fit_transform(queries)
        self.model.fit(X, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, query: str) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Train or load a model before inference")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        probs = self.model.predict_proba(self.vectorizer.transform([query]))[0]
        # The model.classes_ order, not self.labels, defines predict_proba ordering.
        return {str(label): float(p) for label, p in zip(self.model.classes_, probs)}

    def predict(self, query: str) -> tuple[str, float]:
        probs = self.predict_proba(query)
        intent = max(probs, key=probs.get)
        return intent, probs[intent]

    def analyze(self, query: str, uncertainty_threshold: float = 0.60) -> dict:
        if not 0 <= uncertainty_threshold <= 1:
            raise ValueError("uncertainty_threshold must be in [0,1]")
        probs = self.predict_proba(query)
        intent = max(probs, key=probs.get)
        confidence = probs[intent]
        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "uncertain": confidence < uncertainty_threshold,
            "probabilities": probs,
        }

    def save(self, path: str | Path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "labels": self.labels,
            "config": {
                "max_features": self.max_features,
                "random_state": self.random_state,
                "use_cues": self.use_cues,
                "cue_weight": self.cue_weight,
                "C": self.C,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path):
        # joblib uses pickle: load only models created by a trusted source.
        obj = joblib.load(path)
        inst = cls(**obj.get("config", {}))
        inst.vectorizer = obj["vectorizer"]
        inst.model = obj["model"]
        inst.labels = obj["labels"]
        inst.is_fitted = True
        return inst
