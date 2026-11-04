#!/usr/bin/env python3
"""Week 7: select intent model with train-only grouped CV, then test once.

The outer hold-out consists of unseen query-template groups, with the original
seed and split semantics maintained for comparison with the previous baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from query_intent_classifier import QueryIntentClassifier, LABELS


CANDIDATES = (
    # Keep a plain TF-IDF baseline in the CV search for an honest comparison.
    {"name": "word_tfidf_only", "max_features": 500, "use_cues": False, "cue_weight": 0.0, "C": 1.0},
    {"name": "word_plus_cues_2", "max_features": 5000, "use_cues": True, "cue_weight": 2.0, "C": 1.0},
    {"name": "word_plus_cues_4", "max_features": 5000, "use_cues": True, "cue_weight": 4.0, "C": 1.0},
    {"name": "word_plus_cues_6", "max_features": 5000, "use_cues": True, "cue_weight": 6.0, "C": 1.0},
)


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No labeled queries found")
    needed = {"query", "intent", "template_group"}
    if not needed.issubset(rows[0]):
        raise ValueError(f"Missing columns: {needed - set(rows[0])}")
    for row in rows:
        if row["intent"] not in LABELS or not row["query"].strip() or not row["template_group"].strip():
            raise ValueError(f"Invalid query row: {row}")
    if len({r["query"].casefold().strip() for r in rows}) != len(rows):
        raise ValueError("Duplicate query strings; deduplicate before splitting")
    return rows


def select_candidate(queries, labels, groups, train_idx, seed: int):
    # Hyperparameter selection sees ONLY the training partition.
    tr_queries = [queries[i] for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    tr_groups = [groups[i] for i in train_idx]
    folds = list(GroupKFold(n_splits=4).split(tr_queries, tr_labels, tr_groups))
    results = []
    for candidate in CANDIDATES:
        scores, macro = [], []
        for cv_train, cv_val in folds:
            model = QueryIntentClassifier(
                max_features=candidate["max_features"],
                random_state=seed,
                use_cues=candidate["use_cues"],
                cue_weight=candidate["cue_weight"],
                C=candidate["C"],
            )
            model.train([tr_queries[i] for i in cv_train], [tr_labels[i] for i in cv_train])
            predictions = [model.predict(tr_queries[i])[0] for i in cv_val]
            gold = [tr_labels[i] for i in cv_val]
            scores.append(float(accuracy_score(gold, predictions)))
            macro.append(float(f1_score(gold, predictions, labels=list(LABELS), average="macro", zero_division=0)))
        results.append({**candidate, "cv_mean_accuracy": round(sum(scores)/len(scores), 6),
                        "cv_mean_macro_f1": round(sum(macro)/len(macro), 6),
                        "cv_fold_accuracy": [round(v, 6) for v in scores]})
    # Deterministic tie-break: candidate list order (simpler cue strength first).
    best = max(results, key=lambda r: (r["cv_mean_macro_f1"], r["cv_mean_accuracy"]))
    return best, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/intent_queries_labeled.csv")
    parser.add_argument("--model", default="models/intent_classifier.joblib")
    parser.add_argument("--report", default="reports/week7_intent_metrics.json")
    parser.add_argument("--predictions", default="reports/week7_test_predictions.csv")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("test-size must be between 0 and 1")
    rows = load_rows(args.data)
    queries = [r["query"] for r in rows]
    labels = [r["intent"] for r in rows]
    groups = [r["template_group"] for r in rows]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                                 random_state=args.seed).split(queries, labels, groups))
    if set(groups[i] for i in train_idx) & set(groups[i] for i in test_idx):
        raise AssertionError("Template-group leakage between train and test")
    if set(labels[i] for i in train_idx) != set(LABELS) or set(labels[i] for i in test_idx) != set(LABELS):
        raise ValueError("Group split must include all three classes in both partitions")
    selected, cv_results = select_candidate(queries, labels, groups, train_idx, args.seed)
    model = QueryIntentClassifier(max_features=selected["max_features"], random_state=args.seed,
                                  use_cues=selected["use_cues"], cue_weight=selected["cue_weight"],
                                  C=selected["C"])
    model.train([queries[i] for i in train_idx], [labels[i] for i in train_idx])
    results = [model.analyze(queries[i], args.threshold) for i in test_idx]
    predicted = [r["intent"] for r in results]
    gold = [labels[i] for i in test_idx]
    accuracy = float(accuracy_score(gold, predicted))
    report = {
        "dataset": args.data,
        "source_types": dict(Counter(r.get("source", "unknown") for r in rows)),
        "test_method": "GroupShuffleSplit(template_group) with seed 42 by default; model selected by GroupKFold within train only",
        "train_count": len(train_idx), "test_count": len(test_idx),
        "train_class_counts": dict(Counter(labels[i] for i in train_idx)),
        "test_class_counts": dict(Counter(gold)),
        "train_groups": len({groups[i] for i in train_idx}),
        "test_groups": len({groups[i] for i in test_idx}),
        "model_selection_train_only": cv_results,
        "selected_model": selected,
        "accuracy": accuracy,
        "target_accuracy_0_80_met": bool(accuracy >= .80),
        "classification_report": classification_report(gold, predicted, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=list(LABELS)).tolist(),
        "uncertain_count": sum(r["uncertain"] for r in results),
        "uncertainty_threshold": args.threshold,
        "warning": (
            "All dataset queries are curated synthetic examples, not independently labeled real user queries. "
            "Template-disjoint evaluation limits exact template leakage but does not measure real-world "
            "generalization, actual buying intent, or probability calibration."
        ),
    }
    for path in (args.model, args.report, args.predictions):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(args.predictions, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "query", "gold_intent", "predicted_intent", "confidence", "uncertain", "probabilities_json"])
        writer.writeheader()
        for index, result in zip(test_idx, results):
            writer.writerow({"id": rows[index].get("id", ""), "query": result["query"],
                             "gold_intent": labels[index], "predicted_intent": result["intent"],
                             "confidence": round(result["confidence"], 6), "uncertain": result["uncertain"],
                             "probabilities_json": json.dumps(result["probabilities"])})
    model.save(args.model)  # Saved model is fit on train only; held-out data not seen during training.
    print(f"Selected {selected['name']} via training-only group CV (accuracy {selected['cv_mean_accuracy']:.3f}).")
    print(f"Train/test: {len(train_idx)}/{len(test_idx)} | accuracy: {accuracy:.3%} | target met: {accuracy >= .80}")
    print("Saved", args.model, args.report, args.predictions)


if __name__ == "__main__":
    main()
