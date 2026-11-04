#!/usr/bin/env python3
"""Week 6 signal extraction evaluation using independent gold annotations.

Example:
  python scripts/evaluate_signals.py template --predictions data/processed/listing_signals_full.jsonl --source-csv data/processed/listing_sample_10k.csv --sample 200 --output data/processed/signal_gold_template.jsonl
  # Manually review and edit each gold_... field, setting reviewed=true.
  python scripts/evaluate_signals.py evaluate --predictions data/processed/listing_signals_full.jsonl --gold data/processed/signal_gold_reviewed.jsonl --source-csv data/processed/listing_sample_10k.csv --report reports/week6_accuracy.json

IMPORTANT: Predicted labels are NOT gold labels. Evaluate free text only after independent review.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = ('amenities', 'condition_keywords', 'financing_terms', 'location_features')
FIELDS = ('city', 'bedrooms', 'bathrooms', 'price')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{lineno}: {e}') from e
                if not isinstance(obj, dict):
                    raise ValueError(f'{path}:{lineno}: expected JSON object')
                yield obj


def key(value):
    if value is None or str(value).strip() == '':
        return None
    return str(value).strip()


def load_predictions(path, duplicate_policy='first'):
    """Load one prediction per listing ID, reporting duplicate and conflicting rows.

    The full MLS table may contain multiple rows for a listing. Default: first
    occurrence wins. Use --duplicate-policy error to enforce unique IDs, or
    last to prefer the last occurrence. Conflicts require source-data review.
    """
    result = {}
    duplicates = 0
    conflicts = 0
    conflict_examples = []
    for row in read_jsonl(path):
        listing_id = key(row.get('listing_id'))
        if listing_id is None:
            raise ValueError('Prediction without listing_id')
        if listing_id in result:
            duplicates += 1
            if row != result[listing_id]:
                conflicts += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append(listing_id)
            if duplicate_policy == 'error':
                raise ValueError(f'Duplicate prediction listing_id: {listing_id}')
            if duplicate_policy == 'last':
                result[listing_id] = row
        else:
            result[listing_id] = row
    if duplicates:
        print(f'WARNING: {duplicates:,} duplicate prediction rows across '
              f'{len(result):,} unique listing IDs; policy={duplicate_policy}.')
        if conflicts:
            print(f'WARNING: {conflicts:,} duplicates have different content. '
                  f'Example IDs: {", ".join(conflict_examples)}. '
                  'Review these records; choosing one row may affect accuracy.')
    return result


def load_csv(path):
    rows = {}
    duplicates = 0
    conflicts = 0
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            listing_id = key(row.get('L_ListingID', row.get('listing_id')))
            if listing_id is None:
                continue
            if listing_id in rows:
                duplicates += 1
                if row != rows[listing_id]:
                    conflicts += 1
                continue  # Keep first source record, consistent with predictions.
            rows[listing_id] = row
    if duplicates:
        print(f'WARNING: source CSV has {duplicates:,} duplicate listing-ID rows '
              f'({conflicts:,} with different content); keeping the first.')
    return rows


def canonical_number(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
        return None
    try:
        return float(str(value).replace(',', '').replace('$', ''))
    except (ValueError, TypeError):
        return str(value).strip().casefold()


def canonical(field, value):
    if field == 'city':
        return None if value is None or not str(value).strip() else re.sub(r'\s+', ' ', str(value)).strip().casefold()
    return canonical_number(value)


def normalized_labels(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a JSON list of canonical signal strings')
    return {re.sub(r'\s+', ' ', str(v).strip()).casefold() for v in value if str(v).strip()}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'tp': tp, 'fp': fp, 'fn': fn}


def build_template(args):
    predictions = load_predictions(args.predictions, args.duplicate_policy)
    source = load_csv(args.source_csv) if args.source_csv else {}
    ids = list(predictions)
    if args.sample > 0:
        ids = random.Random(args.seed).sample(ids, min(args.sample, len(ids)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for listing_id in ids:
            pred = predictions[listing_id]
            src = source.get(listing_id, {})
            # Use blank gold lists, NEVER prefill from predictions.
            row = {
                'listing_id': pred['listing_id'],
                'remarks': src.get('L_Remarks') or src.get('remarks') or '',
                'predicted': {cat: pred.get(cat, []) for cat in CATEGORIES},
                'gold': {cat: [] for cat in CATEGORIES},
                'reviewed': False,
                'review_notes': ''
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {len(ids)} annotation rows to {output}')
    if not source:
        print('WARNING: No source CSV supplied; provide original remarks to reviewers separately.')
    else:
        missing = sum(not (source.get(i, {}).get('L_Remarks') or source.get(i, {}).get('remarks')) for i in ids)
        if missing:
            print(f'WARNING: {missing} selected records have no source remarks; review requires remarks from MySQL/original CSV.')
    print('Human review required: fill gold lists from remarks (not predicted lists), set reviewed=true.')


def evaluate(args):
    predictions = load_predictions(args.predictions, args.duplicate_policy)
    reviewed = {}
    skipped_unreviewed = 0
    for row in read_jsonl(args.gold):
        listing_id = key(row.get('listing_id'))
        if listing_id is None or listing_id in reviewed:
            raise ValueError(f'Missing or duplicate gold listing_id: {listing_id}')
        if row.get('reviewed') is not True:
            skipped_unreviewed += 1
            continue
        if not isinstance(row.get('gold'), dict) or any(cat not in row['gold'] for cat in CATEGORIES):
            raise ValueError(f'Reviewed listing {listing_id} must include all four gold category arrays')
        reviewed[listing_id] = row
    source = load_csv(args.source_csv) if args.source_csv else {}
    counters = {cat: Counter() for cat in CATEGORIES}
    structured = {field: Counter() for field in FIELDS}
    errors = []
    match_count = 0
    for listing_id, row in reviewed.items():
        if listing_id not in predictions:
            raise ValueError(f'Gold listing {listing_id} missing from predictions')
        pred = predictions[listing_id]
        match_count += 1
        for cat in CATEGORIES:
            expected = normalized_labels(row['gold'][cat], f'{listing_id}.gold.{cat}')
            actual = normalized_labels(pred.get(cat, []), f'{listing_id}.{cat}')
            tp, fp, fn = actual & expected, actual - expected, expected - actual
            counters[cat].update({'tp': len(tp), 'fp': len(fp), 'fn': len(fn), 'exact': int(actual == expected), 'listings': 1})
            if fp or fn:
                errors.append({'listing_id': listing_id, 'category': cat, 'false_positive': sorted(fp), 'false_negative': sorted(fn)})
        if listing_id in source:
            src = source[listing_id]
            truth_fields = {'city': src.get('L_City', src.get('city')),
                            'bedrooms': src.get('L_Keyword2', src.get('beds')),
                            'bathrooms': src.get('LM_Dec_3', src.get('baths')),
                            'price': src.get('L_SystemPrice', src.get('price'))}
            for field in FIELDS:
                expected = canonical(field, truth_fields[field])
                if expected is None:
                    structured[field]['missing_ground_truth'] += 1
                    continue
                actual = canonical(field, pred.get('structured', {}).get(field))
                structured[field]['evaluated'] += 1
                if actual == expected:
                    structured[field]['correct'] += 1
                else:
                    errors.append({'listing_id': listing_id, 'category': 'structured.' + field,
                                   'expected': truth_fields[field], 'predicted': pred.get('structured', {}).get(field)})

    categories = {}
    aggregate = Counter()
    for cat, counts in counters.items():
        categories[cat] = {**prf(counts['tp'], counts['fp'], counts['fn']),
                           'exact_listing_accuracy': round(counts['exact'] / counts['listings'], 4) if counts['listings'] else None,
                           'evaluated_listings': counts['listings']}
        aggregate.update({k: counts[k] for k in ('tp', 'fp', 'fn')})
    field_metrics = {}
    totals = Counter()
    for field, c in structured.items():
        totals.update({'evaluated': c['evaluated'], 'correct': c['correct']})
        field_metrics[field] = {'correct': c['correct'], 'evaluated': c['evaluated'],
                                'accuracy': round(c['correct'] / c['evaluated'], 4) if c['evaluated'] else None,
                                'missing_ground_truth': c['missing_ground_truth']}
    free_micro = prf(aggregate['tp'], aggregate['fp'], aggregate['fn'])
    report = {
        'predictions_total': len(predictions),
        'gold_reviewed_matched': match_count,
        'gold_unreviewed_skipped': skipped_unreviewed,
        'gold_source': args.gold,
        'structured_comparison_source': args.source_csv,
        'free_text': {'by_category': categories,
                      'micro': free_micro,
                      'macro_f1': round(sum(categories[c]['f1'] for c in CATEGORIES) / len(CATEGORIES), 4) if match_count else None,
                      'target_f1_0_75_met': free_micro['f1'] >= .75 if match_count else None},
        'structured': {'by_field': field_metrics,
                       'micro_field_accuracy': round(totals['correct']/totals['evaluated'], 4) if totals['evaluated'] else None,
                       'target_accuracy_0_90_met': totals['correct']/totals['evaluated'] >= .90 if totals['evaluated'] else None},
        'notes': [
            'Free-text F1 compares canonical signal sets against independently human-reviewed annotations.',
            'Structured accuracy compares output structured fields to original MLS columns, not to entity mentions.',
            'Structured accuracy is not an independent validation of underlying MLS values.',
            'Only IDs with reviewed=true are used for free-text evaluation.',
            'No results are claimed from unreviewed/predicted-only labels.'
        ]
    }
    if not match_count:
        print('WARNING: No reviewed gold rows. Free-text metrics are NOT valid until independent labels exist.')
    if not source:
        print('WARNING: No original source CSV. Structured accuracy will be null (not evaluated).')
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    error_path = Path(args.errors) if args.errors else report_path.with_name(report_path.stem + '_errors.jsonl')
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open('w', encoding='utf-8') as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Saved: {report_path} and {error_path}')


def main():
    p = argparse.ArgumentParser(description='Week 6 signal extraction accuracy evaluator')
    sub = p.add_subparsers(dest='command', required=True)
    template = sub.add_parser('template', help='Generate an unreviewed random annotation sample')
    template.add_argument('--predictions', required=True)
    template.add_argument('--duplicate-policy', choices=['first', 'last', 'error'], default='first', help='How to handle repeated listing IDs')
    template.add_argument('--source-csv')
    template.add_argument('--sample', type=int, default=200, help='0 means every listing')
    template.add_argument('--seed', type=int, default=42)
    template.add_argument('--output', default='data/processed/signal_gold_template.jsonl')
    template.set_defaults(func=build_template)
    evaluation = sub.add_parser('evaluate', help='Evaluate reviewed gold labels and structured MLS fields')
    evaluation.add_argument('--predictions', required=True)
    evaluation.add_argument('--duplicate-policy', choices=['first', 'last', 'error'], default='first', help='How to handle repeated listing IDs')
    evaluation.add_argument('--gold', required=True)
    evaluation.add_argument('--source-csv', help='Original CSV containing MLS fields and listing IDs')
    evaluation.add_argument('--report', default='reports/week6_accuracy.json')
    evaluation.add_argument('--errors')
    evaluation.set_defaults(func=evaluate)
    args = p.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
