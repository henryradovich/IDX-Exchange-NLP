#!/usr/bin/env python3
"""Text cleaning and profiling utilities for MLS remarks and user queries."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import Any

import pandas as pd


class TextCleaner:
    """Normalize MLS remarks and real-estate search queries."""

    def __init__(self) -> None:
        # 40+ mappings. Longer/more-specific forms are intentionally included.
        self.abbrev_map = {
            "br": "bedroom",
            "bdrm": "bedroom",
            "bdrms": "bedrooms",
            "bd": "bedroom",
            "beds": "bedrooms",
            "ba": "bathroom",
            "bth": "bathroom",
            "bths": "bathrooms",
            "baths": "bathrooms",
            "mbr": "primary bedroom",
            "mstr br": "primary bedroom",
            "mstr bdrm": "primary bedroom",
            "mbdrm": "primary bedroom",
            "mbath": "primary bathroom",
            "mstr ba": "primary bathroom",
            "sqft": "square feet",
            "sq ft": "square feet",
            "sf": "square feet",
            "sq. ft.": "square feet",
            "w/": "with",
            "w/o": "without",
            "w/d": "washer dryer",
            "a/c": "air conditioning",
            "ac": "air conditioning",
            "hvac": "heating ventilation air conditioning",
            "fp": "fireplace",
            "fplc": "fireplace",
            "gar": "garage",
            "att gar": "attached garage",
            "det gar": "detached garage",
            "bsmt": "basement",
            "fin bsmt": "finished basement",
            "unfin bsmt": "unfinished basement",
            "lr": "living room",
            "dr": "dining room",
            "fam rm": "family room",
            "fr": "family room",
            "kit": "kitchen",
            "lndry": "laundry",
            "lau": "laundry",
            "hoa": "homeowners association",
            "apt": "apartment",
            "condo": "condominium",
            "yr": "year",
            "yrs": "years",
            "mo": "month",
            "mos": "months",
            "approx": "approximately",
            "incl": "including",
            "excl": "excluding",
            "pkg": "parking",
            "prkg": "parking",
            "st": "street",
            "ave": "avenue",
            "blvd": "boulevard",
            "rd": "road",
            "ct": "court",
            "ln": "lane",
            "hwy": "highway",
            "pkwy": "parkway",
        }

        self._generic_stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "your", "you",
            "are", "was", "were", "has", "have", "had", "not", "but", "all",
            "home", "property", "into", "its", "our", "their", "there", "here",
        }

    def clean_text(self, text: Any) -> str:
        """Run the complete cleaning pipeline in a stable order."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        text = str(text)
        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_whitespace(text)
        return text.strip()

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode punctuation/spacing while preserving readable text."""
        if not isinstance(text, str):
            text = str(text)

        text = html.unescape(text)
        replacements = {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2212": "-",
            "\u00a0": " ", "\u200b": "", "\ufeff": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return unicodedata.normalize("NFKC", text)

    def remove_html(self, text: str) -> str:
        """Remove HTML/XML tags and decode common HTML entities."""
        text = html.unescape(str(text)).replace("\u00a0", " ")
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return text

    def normalize_prices(self, text: str) -> str:
        """Normalize shorthand prices: 450k -> 450000, 1.2m -> 1200000."""
        def repl_m(match: re.Match) -> str:
            prefix = match.group("prefix") or ""
            value = int(float(match.group("num").replace(",", "")) * 1_000_000)
            return f"{prefix}{value}"

        def repl_k(match: re.Match) -> str:
            prefix = match.group("prefix") or ""
            value = int(float(match.group("num").replace(",", "")) * 1_000)
            return f"{prefix}{value}"

        # Require token boundaries so words such as "room" are untouched.
        text = re.sub(
            r"(?<![\w.])(?P<prefix>\$\s*)?(?P<num>\d+(?:\.\d+)?)\s*[mM]\b",
            repl_m,
            text,
        )
        text = re.sub(
            r"(?<![\w.])(?P<prefix>\$\s*)?(?P<num>\d+(?:\.\d+)?)\s*[kK]\b",
            repl_k,
            text,
        )
        return text

    def normalize_measurements(self, text: str) -> str:
        """Normalize square-foot and acre measurements."""
        # 2,000 sqft / 2000 sq ft / 2000 sf -> 2000 square feet
        def sqft_repl(match: re.Match) -> str:
            number = match.group("num").replace(",", "")
            return f"{number} square feet"

        text = re.sub(
            r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|sqft|sf)\b",
            sqft_repl,
            text,
            flags=re.I,
        )

        # 0.5 ac / 2 acres -> 0.5 acres / 2 acres
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*(?:ac|acre)\b",
            r"\1 acres",
            text,
            flags=re.I,
        )

        # Normalize feet/inches symbols when directly following a number.
        text = re.sub(r"(\d+)\s*['′]\b", r"\1 feet", text)
        text = re.sub(r'(\d+)\s*["″]\b', r"\1 inches", text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand known MLS abbreviations without replacing substrings inside words."""
        result = str(text)

        # Longest first prevents "sq ft" from being partially consumed.
        for abbreviation, expansion in sorted(
            self.abbrev_map.items(), key=lambda item: len(item[0]), reverse=True
        ):
            escaped = re.escape(abbreviation)
            # Custom boundaries work better than \b for mappings containing "/" or ".".
            pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
            result = re.sub(pattern, expansion, result, flags=re.I)

        return result

    def normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation without destroying useful numeric characters."""
        text = re.sub(r"[•·▪◦]", " ", text)
        text = re.sub(r"\.{3,}", ".", text)
        text = re.sub(r"([!?.,;:])\1+", r"\1", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Collapse tabs/newlines/repeated spaces into single spaces."""
        return re.sub(r"\s+", " ", str(text)).strip()

    def profile_column(self, df: pd.DataFrame, column_name: str) -> dict[str, Any]:
        """Profile a text column before cleaning."""
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found.")

        series = df[column_name]
        text = series.fillna("").astype(str)

        return {
            "row_count": int(len(series)),
            "null_rate": float(series.isnull().mean()),
            "avg_length": float(text.str.len().mean()),
            "median_length": float(text.str.len().median()),
            "max_length": int(text.str.len().max()) if len(text) else 0,
            "price_mentions": int(text.str.contains(r"\$\s*\d", regex=True).sum()),
            "shorthand_price_mentions": int(
                text.str.contains(r"(?i)(?:\$?\s*\d+(?:\.\d+)?)\s*[km]\b", regex=True).sum()
            ),
            "html_presence": int(text.str.contains(r"<[^>]+>", regex=True).sum()),
            "non_ascii_rows": int(text.map(lambda x: any(ord(ch) > 127 for ch in x)).sum()),
            "measurement_mentions": int(
                text.str.contains(
                    r"(?i)\d[\d,]*(?:\.\d+)?\s*(?:sq\.?\s*ft\.?|sqft|sf|acres?|ac)\b",
                    regex=True,
                ).sum()
            ),
            "common_abbreviations": self._detect_abbreviations(text),
            "common_terms": self._extract_top_ngrams(text),
        }

    def _detect_abbreviations(self, series: pd.Series, top_n: int = 30) -> list[dict[str, Any]]:
        found = Counter()
        for abbreviation in self.abbrev_map:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(abbreviation)}(?![A-Za-z0-9])",
                flags=re.I,
            )
            count = sum(len(pattern.findall(text)) for text in series)
            if count:
                found[abbreviation] = count

        return [
            {"abbreviation": key, "count": int(count)}
            for key, count in found.most_common(top_n)
        ]

    def _extract_top_ngrams(
        self, series: pd.Series, n: int = 2, top_n: int = 25
    ) -> list[dict[str, Any]]:
        counts = Counter()

        for value in series:
            tokens = re.findall(r"[a-z]+", str(value).lower())
            tokens = [t for t in tokens if len(t) > 2 and t not in self._generic_stopwords]

            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                counts[phrase] += 1

        return [
            {"term": term, "count": int(count)}
            for term, count in counts.most_common(top_n)
        ]


def clean_dataframe(
    input_path: str = "data/processed/listing_sample.csv",
    output_path: str = "data/processed/listing_sample_cleaned.csv",
) -> pd.DataFrame:
    """Clean the remarks column and save a new dataset."""
    df = pd.read_csv(input_path)
    cleaner = TextCleaner()

    if "remarks" not in df.columns:
        raise ValueError("Input dataset must contain a 'remarks' column.")

    df["remarks_original"] = df["remarks"]
    df["remarks_cleaned"] = df["remarks"].map(cleaner.clean_text)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cleaned = clean_dataframe()
    print(f"Saved {len(cleaned)} cleaned rows to data/processed/listing_sample_cleaned.csv")
