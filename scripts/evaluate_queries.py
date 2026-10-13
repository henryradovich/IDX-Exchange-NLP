#!/usr/bin/env python3
"""Compare query parsing against independently specified expected filters.

Run from repository root:
    python3 scripts/evaluate_queries.py

The reference file is created independently of QueryParser and should be reviewed
by teammates. This reports exact structured-filter accuracy, not intent accuracy.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from query_parser import QueryParser


def canonical(value):
    """Ignore ordering of sets represented as lists, not mismatched values."""
    if isinstance(value, list):
        return sorted(set(str(item).lower().strip() for item in value))
    if isinstance(value, str):
        return value.strip().lower()
    return value


def compare(expected, predicted):
    expected = {k: canonical(v) for k, v in expected.items()}
    predicted = {k: canonical(v) for k, v in predicted.items()}
    missing = {k: v for k, v in expected.items() if k not in predicted}
    unexpected = {k: v for k, v in predicted.items() if k not in expected}
    wrong = {k: {'expected': expected[k], 'actual': predicted[k]}
             for k in expected.keys() & predicted.keys() if expected[k] != predicted[k]}
    return not (missing or unexpected or wrong), missing, unexpected, wrong


def evaluate(args):
    gold_file = Path(args.gold)
    if not gold_file.exists():
        raise FileNotFoundError(f'Missing reference file: {gold_file}')
    rows = json.loads(gold_file.read_text(encoding='utf-8'))['queries']
    parser = QueryParser(args.taxonomy)
    by_intent = defaultdict(lambda: {'correct': 0, 'total': 0})
    results = []
    for row in rows:
        expected = row['expected_filters']
        actual = parser.parse(row['query'])
        correct, missing, extra, wrong = compare(expected, actual)
        by_intent[row['intent']]['total'] += 1
        by_intent[row['intent']]['correct'] += int(correct)
        results.append({
            'id': row['id'], 'query': row['query'], 'intent': row['intent'],
            'exact_match': correct,
            'expected_filters': expected, 'predicted_filters': actual,
            'missing_filters': missing, 'unexpected_filters': extra,
            'wrong_filters': wrong,
        })

    correct_count = sum(int(x['exact_match']) for x in results)
    total = len(results)
    accuracy = correct_count / total if total else 0.0
    report = {
        'evaluation_type': 'exact structured-filter match (all expected keys and no extras)',
        'total_queries': total,
        'correct_queries': correct_count,
        'incorrect_queries': total - correct_count,
        'accuracy': round(accuracy, 4),
        'target_accuracy': 0.90,
        'target_met': accuracy >= 0.90,
        'by_intent': {
            label: {**counts, 'accuracy': round(counts['correct'] / counts['total'], 4)}
            for label, counts in sorted(by_intent.items())
        },
        'limitations': [
            'Sample queries were generated for the project; performance here is in-sample development-set evaluation, not independent generalization.',
            'Reference filters require human/team review before they are treated as gold.',
            'SQL execution and whether each supported filter is actually applied in SQL are separate from parsing accuracy.',
            'Taxonomy coverage and synonym choices influence the score.',
        ],
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'query_accuracy_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (outdir/'query_predictions.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (outdir/'query_error_analysis.md').open('w', encoding='utf-8') as f:
        f.write(f'# Week 4 query parsing error analysis\n\nExact match: **{correct_count}/{total} ({accuracy:.1%})**. Target: 90%.\n\n')
        f.write('## Accuracy by intent\n\n| Intent | Correct | Total | Accuracy |\n|---|---:|---:|---:|\n')
        for label, counts in sorted(by_intent.items()):
            f.write(f'| {label} | {counts["correct"]} | {counts["total"]} | {counts["correct"]/counts["total"]:.1%} |\n')
        f.write('\n## Incorrect parses\n\n')
        for r in results:
            if not r['exact_match']:
                f.write(f'### {r["id"]}. {r["query"]}\n\n')
                f.write('Expected: `' + json.dumps(r['expected_filters']) + '`\n\n')
                f.write('Actual: `' + json.dumps(r['predicted_filters']) + '`\n\n')
                f.write('Missing: `' + json.dumps(r['missing_filters']) + '`; Extra: `' + json.dumps(r['unexpected_filters']) + '`; Wrong: `' + json.dumps(r['wrong_filters']) + '`\n\n')
    print(f'Exact-match query accuracy: {correct_count}/{total} = {accuracy:.2%}')
    print('90% target:', 'MET' if report['target_met'] else 'NOT MET')
    print('Files written to:', outdir)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', default='data/processed/sample_queries_gold_filters.json')
    parser.add_argument('--taxonomy', default='data/processed/taxonomy.json')
    parser.add_argument('--outdir', default='reports')
    evaluate(parser.parse_args())
