#!/usr/bin/env python3
"""Create a silver-labeled entity dataset from cleaned Week 2 remarks.

IMPORTANT: These labels are generated automatically and should be manually reviewed
before they are treated as gold labels for final F1 evaluation.
"""

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from entity_extractor import EntityExtractor


INPUT_PATH = Path("data/processed/listing_sample_cleaned.csv")
FALLBACK_INPUT_PATH = Path("data/processed/listing_sample.csv")
OUTPUT_PATH = Path("data/processed/entity_labels.jsonl")
SAMPLE_SIZE = 250
RANDOM_STATE = 42


def main():
    input_path = INPUT_PATH if INPUT_PATH.exists() else FALLBACK_INPUT_PATH
    if not input_path.exists():
        raise FileNotFoundError(
            "Could not find listing_sample_cleaned.csv or listing_sample.csv."
        )

    df = pd.read_csv(input_path)

    text_col = "remarks_cleaned" if "remarks_cleaned" in df.columns else "remarks"
    if text_col not in df.columns:
        raise ValueError("Dataset must contain remarks or remarks_cleaned.")

    extractor = EntityExtractor()

    # Stable 250-row sample.
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            record = {
                "listing_id": row.get("L_ListingID"),
                "text": text,
                "entities": extractor.extract_spans(text),
                "annotation_source": "rule_based_silver",
                "needs_manual_review": True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} silver-labeled remarks to {OUTPUT_PATH}")
    print("Manual review is required before using this as a true gold evaluation set.")


if __name__ == "__main__":
    main()
