#!/usr/bin/env python3
"""Evaluate entity spans with exact-match precision, recall, and F1."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from entity_extractor import EntityExtractor


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def as_key(entity):
    return (int(entity["start"]), int(entity["end"]), str(entity["label"]))


def safe_div(a, b):
    return a / b if b else 0.0


def evaluate(path="data/processed/entity_labels_corrected.jsonl"):
    extractor = EntityExtractor()

    overall = Counter()
    by_label = defaultdict(Counter)
    errors = []

    for record in load_jsonl(path):
        text = record["text"]
        gold = {as_key(e) for e in record.get("entities", [])}
        pred_entities = extractor.extract_spans(text)
        pred = {as_key(e) for e in pred_entities}

        tp = gold & pred
        fp = pred - gold
        fn = gold - pred

        overall["tp"] += len(tp)
        overall["fp"] += len(fp)
        overall["fn"] += len(fn)

        labels = {x[2] for x in gold | pred}
        for label in labels:
            g = {x for x in gold if x[2] == label}
            p = {x for x in pred if x[2] == label}
            by_label[label]["tp"] += len(g & p)
            by_label[label]["fp"] += len(p - g)
            by_label[label]["fn"] += len(g - p)

        for item in sorted(fp):
            errors.append({
                "type": "false_positive",
                "label": item[2],
                "start": item[0],
                "end": item[1],
                "span": text[item[0]:item[1]],
                "text": text,
            })
        for item in sorted(fn):
            errors.append({
                "type": "false_negative",
                "label": item[2],
                "start": item[0],
                "end": item[1],
                "span": text[item[0]:item[1]],
                "text": text,
            })

    precision = safe_div(overall["tp"], overall["tp"] + overall["fp"])
    recall = safe_div(overall["tp"], overall["tp"] + overall["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)

    report = {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            **dict(overall),
        },
        "by_label": {},
    }

    for label, counts in sorted(by_label.items()):
        p = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        r = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        label_f1 = safe_div(2 * p * r, p + r)
        report["by_label"][label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(label_f1, 4),
            **dict(counts),
        }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/entity_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open("reports/entity_errors.jsonl", "w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2))

    # Do not claim the 85% target if the labels are still silver.
    print(
        "\nImportant: if entity_labels.jsonl still contains "
        "'annotation_source': 'rule_based_silver', this score is not a valid "
        "independent estimate of model quality. Manually review/correct the spans first."
    )


if __name__ == "__main__":
    evaluate()
