"""Week 6 listing signals, with expanded, canonical amenity matching.

Drop-in replacement for scripts/signal_extractor.py. No new dependencies.
Avoids counting negated/hypothetical amenities as existing features.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

CATEGORY_OUTPUT = {
    'Amenities': 'amenities', 'Property Features': 'amenities',
    'Exterior': 'amenities', 'Condition': 'condition_keywords',
    'Location': 'location_features', 'Pricing & Transaction': 'financing_terms',
}
# Only explicitly useful rooms/areas: don't turn generic words like bedroom/kitchen
# into amenity signals.
ROOM_AMENITIES = {
    'garage', 'walk in closet', 'laundry room', 'pantry', 'workshop',
    'home office', 'mudroom', 'walk in pantry', 'sunroom', 'media room',
    'game room', 'bonus room', 'guest suite', 'loft', 'exercise room',
}
FINANCING_TERMS = {'seller financing', 'cash offer', 'closing costs'}
EXTRA_FINANCING = {
    'owner financing': 'seller financing',
    'seller will carry': 'seller financing',
    'assumable loan': 'assumable loan',
    'fha financing': 'fha financing',
    'va financing': 'va financing',
}
EXCLUDE_AMENITY = {'hoa', 'floor plan', 'built in', 'outdoor living', 'community amenities',
                   'outdoor space', 'electric vehicle', 'stainless steel'}
# Canonical terms are deliberately comparable to the current Week 6 gold candidate labels.
# Specific original phrases remain available in `matches` as `text` and `source_term`.
AMENITY_CANONICAL = {
    'swimming pool': 'pool', 'community pool': 'pool',
    'car garage': 'garage', 'two car garage': 'garage',
    'three car garage': 'garage', 'attached garage': 'garage',
    'landscaped': 'landscaping', 'wood floors': 'hardwood floors',
    'built ins': 'built ins', 'built-in': 'built ins',
    'vaulted ceilings': 'high ceilings',
    'open concept': 'open floor plan',
    'private backyard': 'backyard', 'large backyard': 'backyard',
    'updated kitchen': 'remodeled kitchen',
    'central air': 'air conditioning',
    'shed': 'storage shed',
}
# Conservative lexical aliases: phrase must be explicitly present in the remark.
AMENITY_ALIASES = {
    'garage': ['garages', 'attached garage', 'detached garage', '2 car garage',
               '2-car garage', '3 car garage', 'three-car garage', 'two-car garage'],
    'walk in closet': ['walk-in closet', 'walk in closets', 'walk-in closets',
                       'walkin closet', 'walk-in wardrobe'],
    'laundry room': ['laundry area', 'laundry space', 'laundry room', 'utility laundry room'],
    'pantry': ['kitchen pantry', 'butler pantry', 'butlers pantry', 'butler\'s pantry'],
    'built ins': ['built-ins', 'built in shelving', 'built-in shelving',
                  'built in bookcases', 'built-in bookcases', 'builtin shelves',
                  'custom built-ins'],
    'high ceilings': ['high ceiling', 'tall ceilings', 'vaulted ceiling', 'vaulted ceilings',
                      'soaring ceilings', 'cathedral ceilings'],
    'pool': ['swimming pool', 'community pool', 'private pool', 'in ground pool',
             'in-ground pool', 'inground pool', 'heated pool'],
    'hot tub': ['hot-tub', 'hot tub', 'hot tubs', 'spa tub', 'jacuzzi'],
    'open floor plan': ['open-floor plan', 'open concept', 'open-concept',
                        'open layout', 'open floorplan'],
    'landscaping': ['landscaped', 'professionally landscaped', 'landscape design'],
    'guest house': ['guesthouse', 'guest cottage', 'casita', 'guest quarters'],
    'updated bathroom': ['updated bathrooms', 'remodeled bathroom',
                          'renovated bathroom', 'bathroom remodel', 'remodeled baths'],
    'fitness center': ['fitness centre', 'gym', 'fitness room', 'exercise facility'],
    'hardwood floors': ['hardwood flooring', 'wood flooring', 'wood floors',
                        'hardwood floor', 'oak floors'],
    'air conditioning': ['a/c', 'central ac', 'central a/c', 'central air',
                         'air conditioned'],
    'remodeled kitchen': ['renovated kitchen', 'updated kitchen', 'kitchen remodel',
                          'newly remodeled kitchen'],
    'kitchen island': ['center island', 'centre island', 'island kitchen'],
    'skylights': ['skylight', 'sky lights'],
    'barn': ['horse barn', 'storage barn'],
    'pond': ['private pond', 'stocked pond'],
    'fireplace': ['fire place', 'gas fireplace', 'wood burning fireplace'],
    'stainless steel appliances': ['stainless appliances', 'ss appliances',
                                   'stainless-steel appliances'],
    'ev charging': ['ev charger', 'electric vehicle charger', 'ev charging station',
                    'electric car charging'],
    'sprinkler system': ['irrigation system', 'automatic sprinklers'],
    'tennis court': ['tennis courts'],
    'solar panels': ['solar panel', 'rooftop solar'],
    'fenced yard': ['fenced backyard', 'fully fenced yard'],
    'new windows': ['newly installed windows', 'replacement windows'],
}
# Common claims about absent/future amenities. Favor precision over aggressive matches.
NEGATED_CONTEXT = re.compile(
    r'(?:\bno\b|\bwithout\b|\black(?:s|ing)?\b|\bdoes\s+not\s+(?:have|include)\b|'
    r'\bnot\s+(?:equipped\s+with|included|available)\b)'
    r'(?:\s+\w+){0,3}\s*$', re.I)
FUTURE_CONTEXT = re.compile(
    r'(?:\bpotential\s+for\b|\broom\s+for\b|\bspace\s+for\b|'
    r'\boption\s+(?:to\s+add|for)\b|\bplans?\s+for\b|\bpossible\b|'
    r'\bcan\s+(?:add|install|build)\b|\bready\s+for\b)'
    r'(?:\s+\w+){0,3}\s*$', re.I)
NEGATED_POST = re.compile(r'^\s+(?:not\s+included|not\s+installed|is\s+possible|could\s+be\s+added)\b', re.I)


def _text(value: Any) -> str:
    return '' if value is None else str(value)


def _pattern(term: str) -> re.Pattern:
    # Allow separators, possessives and common punctuation between words.
    words = re.findall(r'[a-z0-9]+', term.lower())
    if not words:
        raise ValueError('Empty signal pattern')
    return re.compile(r'(?<!\w)' + r'[\s\-–/]+'.join(map(re.escape, words)) + r'(?!\w)', re.I)


class SignalExtractor:
    def __init__(self, taxonomy: dict | str | Path, entity_extractor=None):
        if isinstance(taxonomy, (str, Path)):
            with open(taxonomy, encoding='utf-8') as stream:
                taxonomy = json.load(stream)
        self.taxonomy = taxonomy
        self.extractor = entity_extractor
        patterns = []
        existing = set()

        def add(canonical: str, group: str, phrase: str):
            key = (canonical, group, phrase.lower())
            if key in existing:
                return
            existing.add(key)
            patterns.append((canonical, group, phrase, _pattern(phrase)))

        for obj in taxonomy.get('terms', []):
            term = str(obj.get('term', '')).strip().lower()
            category = obj.get('category')
            if not term:
                continue
            if category in ('Interior', 'Rooms') and term in ROOM_AMENITIES:
                group = 'amenities'
            elif category in CATEGORY_OUTPUT:
                group = CATEGORY_OUTPUT[category]
            else:
                continue
            if category == 'Pricing & Transaction' and term not in FINANCING_TERMS:
                continue
            if group == 'amenities' and term in EXCLUDE_AMENITY:
                continue
            canonical = AMENITY_CANONICAL.get(term, term) if group == 'amenities' else term
            add(canonical, group, term)

        # Explicit useful amenities absent from original 200-term taxonomy.
        for canonical, aliases in AMENITY_ALIASES.items():
            add(canonical, 'amenities', canonical)
            for alias in aliases:
                add(canonical, 'amenities', alias)
        for alias, canonical in EXTRA_FINANCING.items():
            add(canonical, 'financing_terms', alias)
        # Check longest matched phrase first, regardless of canonical label length.
        self.terms = sorted(patterns, key=lambda x: (-len(x[2]), x[1], x[0], x[2]))

    def _match(self, remarks: str) -> tuple[dict, list]:
        found = {c: [] for c in set(CATEGORY_OUTPUT.values())}
        matches, occupied = [], []
        for term, group, source_term, pattern in self.terms:
            for match in pattern.finditer(remarks):
                a, b = match.span()
                if any(group == g and a < end and b > start for start, end, g in occupied):
                    continue
                before = remarks[max(0, a - 65):a]
                # Don't inherit a qualifier from an earlier sentence or clause.
                before = re.split(r'[.!?;\n]', before)[-1]
                after = remarks[b:b + 35]
                excluded = (bool(NEGATED_CONTEXT.search(before)) or
                            bool(FUTURE_CONTEXT.search(before)) or
                            bool(NEGATED_POST.match(after)))
                matches.append({'start': a, 'end': b, 'text': remarks[a:b],
                                'term': term, 'source_term': source_term,
                                'category': group, 'negated': excluded})
                occupied.append((a, b, group))
                if not excluded and term not in found[group]:
                    found[group].append(term)
        return {key: sorted(value) for key, value in found.items()}, sorted(matches, key=lambda x: x['start'])

    def extract_signals(self, listing_record: dict) -> dict:
        remarks = _text(listing_record.get('L_Remarks', listing_record.get('remarks_cleaned', listing_record.get('remarks'))))
        grouped, matches = self._match(remarks)
        entities = self.extractor.extract_all(remarks) if self.extractor is not None else {}
        return {
            'listing_id': listing_record.get('L_ListingID', listing_record.get('listing_id')),
            'structured': {
                'city': listing_record.get('L_City', listing_record.get('city')),
                'bedrooms': listing_record.get('L_Keyword2', listing_record.get('beds')),
                'bathrooms': listing_record.get('LM_Dec_3', listing_record.get('baths')),
                'price': listing_record.get('L_SystemPrice', listing_record.get('price')),
            },
            'entities': entities,
            'amenities': grouped['amenities'],
            'condition_keywords': grouped['condition_keywords'],
            'financing_terms': grouped['financing_terms'],
            'location_features': grouped['location_features'],
            'keywords': sorted(set(sum(grouped.values(), []))),
            'matches': matches,
        }
