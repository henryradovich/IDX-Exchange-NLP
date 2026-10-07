#!/usr/bin/env python3
"""Rule-based named entity extraction for cleaned MLS remarks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class EntityExtractor:
    """Extract structured real-estate entities from cleaned text."""

    def __init__(self, taxonomy_path: str = "data/processed/taxonomy.json") -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self.amenity_terms = self._load_amenities()

        self.bedroom_patterns = [
            re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:bedroom|bedrooms|bed|beds|br|bd)\b", re.I),
            re.compile(r"\b(\d+)\s*[- ]?bed(?:room)?\b", re.I),
        ]
        self.bathroom_patterns = [
            re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:bathroom|bathrooms|bath|baths|ba)\b", re.I),
            re.compile(r"\b(\d+(?:\.\d+)?)\s*[- ]?bath(?:room)?\b", re.I),
        ]
        self.sqft_patterns = [
            re.compile(
                r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:square feet|square foot|sq\.?\s*ft\.?|sqft|sf)\b",
                re.I,
            )
        ]
        self.price_patterns = [
            # Cleaned Week 2 text should already expand 450k / 1.2m.
            re.compile(r"\$\s*([\d,]{5,})\b"),
            re.compile(r"\b(?:priced?|price|listed|asking)\s+(?:at|for)?\s*\$?\s*([\d,]{5,})\b", re.I),
        ]

    def _load_amenities(self) -> list[str]:
        """Load amenity phrases from the Week 1 taxonomy."""
        if not self.taxonomy_path.exists():
            return []

        try:
            with self.taxonomy_path.open("r", encoding="utf-8") as f:
                taxonomy = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        terms = []
        for item in taxonomy.get("terms", []):
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip().lower()
            category = str(item.get("category", "")).strip().lower()
            # Use explicit amenities plus useful feature/exterior phrases.
            if term and category in {"amenities", "property features", "exterior"}:
                terms.append(term)

        # Prefer longer phrases so "community pool" is found before "pool".
        return sorted(set(terms), key=lambda x: (-len(x), x))

    @staticmethod
    def _numeric_value(raw: str) -> int | float:
        value = float(raw.replace(",", ""))
        return int(value) if value.is_integer() else value

    def _extract_first_numeric(self, text: str, patterns: list[re.Pattern]) -> int | float | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return self._numeric_value(match.group(1))
        return None

    def extract_bedrooms(self, text: str) -> int | float | None:
        return self._extract_first_numeric(str(text), self.bedroom_patterns)

    def extract_bathrooms(self, text: str) -> int | float | None:
        return self._extract_first_numeric(str(text), self.bathroom_patterns)

    def extract_price(self, text: str) -> int | None:
        for pattern in self.price_patterns:
            match = pattern.search(str(text))
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    def extract_sqft(self, text: str) -> int | float | None:
        return self._extract_first_numeric(str(text), self.sqft_patterns)

    def extract_amenities(self, text: str) -> list[str]:
        """Return taxonomy amenities/features mentioned in the text."""
        lower = str(text).lower()
        found = []
        occupied = []

        for term in self.amenity_terms:
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)
            for match in pattern.finditer(lower):
                # Avoid returning nested duplicates such as both pool and community pool.
                span = match.span()
                if any(span[0] >= a and span[1] <= b for a, b in occupied):
                    continue
                found.append(term)
                occupied.append(span)
                break

        return found

    def extract_spans(self, text: str) -> list[dict[str, Any]]:
        """Extract all matching entity spans for annotation/evaluation."""
        text = str(text)
        entities: list[dict[str, Any]] = []

        def add_numeric(patterns, label):
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append({
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                        "text": match.group(0),
                        "value": self._numeric_value(match.group(1)),
                    })

        add_numeric(self.bedroom_patterns, "BEDROOMS")
        add_numeric(self.bathroom_patterns, "BATHROOMS")
        add_numeric(self.sqft_patterns, "SQFT")

        for pattern in self.price_patterns:
            for match in pattern.finditer(text):
                entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": "PRICE",
                    "text": match.group(0),
                    "value": int(match.group(1).replace(",", "")),
                })

        lower = text.lower()
        occupied_amenities = []
        for term in self.amenity_terms:
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)
            for match in pattern.finditer(lower):
                span = match.span()
                if any(span[0] >= a and span[1] <= b for a, b in occupied_amenities):
                    continue
                entities.append({
                    "start": span[0],
                    "end": span[1],
                    "label": "AMENITY",
                    "text": text[span[0]:span[1]],
                    "value": term,
                })
                occupied_amenities.append(span)

        # Deduplicate exact spans/labels.
        unique = {}
        for entity in entities:
            key = (entity["start"], entity["end"], entity["label"])
            unique[key] = entity

        return sorted(unique.values(), key=lambda e: (e["start"], e["end"], e["label"]))

    def extract_all(self, text: str) -> dict[str, Any]:
        return {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "price": self.extract_price(text),
            "sqft": self.extract_sqft(text),
            "amenities": self.extract_amenities(text),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract entities from MLS text.")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--taxonomy", default="data/processed/taxonomy.json")
    args = parser.parse_args()

    extractor = EntityExtractor(args.taxonomy)
    if args.text:
        print(json.dumps(extractor.extract_all(args.text), indent=2))
