#!/usr/bin/env python3
"""Build a categorized real-estate taxonomy from listing_sample.csv."""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/listing_sample.csv")
OUTPUT_PATH = Path("data/processed/taxonomy.json")
CANDIDATES_PATH = Path("data/processed/taxonomy_candidates.csv")
MIN_TERMS = 200

# Eight categories required by the Week 1 deliverable.
# These are candidate domain concepts; only phrases actually found in the
# listing sample are emitted first. If fewer than MIN_TERMS match, high-
# frequency domain-like n-grams from the sample are added as "Other".
CATEGORY_TERMS = {
    "Property Features": [
        "fireplace", "natural light", "open floor plan", "floor plan", "open concept",
        "high ceilings", "vaulted ceilings", "hardwood floors", "wood floors",
        "tile floors", "new flooring", "granite countertops", "quartz countertops",
        "stainless steel appliances", "stainless steel", "kitchen island",
        "breakfast bar", "built in", "built ins", "custom cabinetry", "updated kitchen",
        "remodeled kitchen", "new appliances", "central air", "air conditioning",
        "forced air", "solar panels", "double pane windows", "new windows", "new roof",
    ],
    "Location": [
        "mountain view", "mountain views", "city view", "city views", "waterfront",
        "lakefront", "golf course", "cul de sac", "corner lot", "near downtown",
        "downtown", "near schools", "near shopping", "near restaurants", "near parks",
        "near transit", "public transportation", "quiet neighborhood", "gated community",
        "shopping dining", "conveniently located", "prime location", "close to",
        "easy access", "walking distance",
    ],
    "Interior": [
        "living room", "living space", "dining area", "dining room", "great room",
        "family room", "primary suite", "primary bedroom", "master bedroom",
        "home office", "guest suite", "bonus room", "laundry room", "mudroom",
        "pantry", "walk in pantry", "walk in closet", "primary bath", "primary bathroom",
        "full bath", "half bath", "powder room", "eat in kitchen", "breakfast nook",
        "main level", "upper level", "lower level",
    ],
    "Exterior": [
        "private backyard", "large backyard", "backyard", "front yard", "fenced yard",
        "covered patio", "patio", "deck", "balcony", "front porch", "covered porch",
        "garden", "mature trees", "landscaping", "landscaped", "sprinkler system",
        "outdoor living", "outdoor space", "fire pit", "hot tub", "pool",
        "attached garage", "two car garage", "three car garage", "car garage",
        "storage shed", "shed",
    ],
    "Amenities": [
        "community pool", "swimming pool", "fitness center", "clubhouse", "tennis court",
        "pickleball", "dog park", "walking trails", "bike trails", "community garden",
        "security system", "smart home", "smart thermostat", "ev charging",
        "electric vehicle", "home theater", "wine cellar", "sauna", "elevator",
        "community amenities", "hoa", "playground", "recreation center",
    ],
    "Rooms": [
        "bedroom", "bedrooms", "bathroom", "bathrooms", "kitchen", "garage",
        "basement", "finished basement", "unfinished basement", "loft", "den", "study",
        "office", "sunroom", "workshop", "storage room", "utility room", "laundry",
        "exercise room", "media room", "game room", "recreation room", "guest room",
        "primary bedroom", "primary suite", "dining room", "living room",
    ],
    "Condition": [
        "new construction", "newly built", "newly remodeled", "recently remodeled",
        "fully remodeled", "renovated", "updated", "move in ready", "turnkey",
        "well maintained", "meticulously maintained", "excellent condition",
        "good condition", "needs updating", "fixer upper", "as is", "historic",
        "restored", "fresh paint", "new paint", "new carpet", "new flooring",
        "new roof", "new windows", "updated bathroom", "updated kitchen",
    ],
    "Pricing & Transaction": [
        "asking price", "list price", "listing price", "price reduction", "reduced price",
        "below market", "market value", "property taxes", "hoa fee", "hoa fees",
        "seller financing", "cash offer", "closing costs", "contingent", "pending",
        "under contract", "active listing", "new listing", "back on market",
        "recently sold", "sold", "investment opportunity", "rare opportunity",
    ],
}

# Phrases like these are common prose, not useful taxonomy concepts.
STOPWORDS = {
    "a","an","and","are","as","at","be","been","by","for","from","has","have","in",
    "is","it","its","of","on","or","that","the","this","to","with","you","your",
    "home","property"
}
GENERIC_PHRASES = {
    "welcome to", "this home", "the home", "the property", "home offers",
    "offers a", "features a", "perfect for", "ideal for", "this is",
    "opportunity to", "a rare", "one of", "plenty of", "designed for",
}


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_frequency(remarks, phrase: str) -> int:
    pattern = re.compile(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)")
    return sum(len(pattern.findall(text)) for text in remarks)


def generate_ngram_candidates(remarks):
    """Return useful frequent 2- and 3-word phrases observed in the sample."""
    counts = Counter()

    for remark in remarks:
        tokens = re.findall(r"[a-z]+", remark)
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                words = tokens[i:i+n]
                phrase = " ".join(words)

                # Reject phrases dominated by grammatical filler.
                if words[0] in STOPWORDS or words[-1] in STOPWORDS:
                    continue
                if phrase in GENERIC_PHRASES:
                    continue
                if sum(w in STOPWORDS for w in words) > 1:
                    continue

                counts[phrase] += 1

    return counts


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run scripts/data_loading.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    if "remarks" not in df.columns:
        raise ValueError("listing_sample.csv must contain a 'remarks' column.")

    remarks = [normalize(x) for x in df["remarks"].dropna() if str(x).strip()]
    if not remarks:
        raise ValueError("No non-empty listing remarks were found.")

    terms = []
    seen = set()

    # Add curated real-estate concepts that actually occur in this dataset.
    for category, candidates in CATEGORY_TERMS.items():
        for candidate in candidates:
            normalized = normalize(candidate)
            if normalized in seen:
                continue

            count = phrase_frequency(remarks, normalized)
            if count > 0:
                seen.add(normalized)
                terms.append({
                    "term": normalized,
                    "category": category,
                    "count": count,
                    "source": "category_match",
                })

    # Supplement with frequent observed n-grams until the taxonomy has 200+ terms.
    ngram_counts = generate_ngram_candidates(remarks)
    for phrase, count in ngram_counts.most_common():
        if len(terms) >= MIN_TERMS:
            break
        if phrase in seen:
            continue
        if count < 2:
            break

        seen.add(phrase)
        terms.append({
            "term": phrase,
            "category": "Other",
            "count": count,
            "source": "sample_ngram",
        })

    # Highest-frequency terms first, then assign stable IDs.
    terms.sort(key=lambda x: (-x["count"], x["term"]))
    for i, term in enumerate(terms, start=1):
        term["id"] = i

    # Coverage = percentage of remarks containing at least one taxonomy phrase.
    taxonomy_phrases = [t["term"] for t in terms]
    matched = 0
    for remark in remarks:
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", remark)
               for term in taxonomy_phrases):
            matched += 1

    coverage = matched / len(remarks) if remarks else 0.0

    category_counts = Counter(t["category"] for t in terms)

    taxonomy = {
        "version": "1.0",
        "source": str(INPUT_PATH),
        "listing_count": len(df),
        "remarks_analyzed": len(remarks),
        "term_count": len(terms),
        "coverage": round(coverage, 4),
        "coverage_percent": round(coverage * 100, 2),
        "categories": dict(sorted(category_counts.items())),
        "terms": terms,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    # Keep the ranked candidates available for manual taxonomy review.
    candidate_rows = [
        {"term": phrase, "frequency": count}
        for phrase, count in ngram_counts.most_common(500)
    ]
    pd.DataFrame(candidate_rows).to_csv(CANDIDATES_PATH, index=False)

    print(f"Analyzed {len(remarks)} remarks")
    print(f"Created {len(terms)} taxonomy terms")
    print(f"Coverage: {coverage:.1%}")
    print("Terms by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    print(f"Saved taxonomy to {OUTPUT_PATH}")
    print(f"Saved candidates to {CANDIDATES_PATH}")

    if len(terms) < MIN_TERMS:
        raise RuntimeError(
            f"Only {len(terms)} taxonomy terms were generated; "
            f"at least {MIN_TERMS} are required."
        )


if __name__ == "__main__":
    main()
