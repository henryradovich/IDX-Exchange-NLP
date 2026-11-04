#!/usr/bin/env python3
"""Combine linguistic intent with Week 4 filters, without equating them."""
import argparse
import json
import sys
from pathlib import Path
from query_intent_classifier import QueryIntentClassifier


def get_week4_parser():
    # Both source scripts sit in scripts/, no __init__.py needed.
    script_dir=Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:sys.path.insert(0,str(script_dir))
    try:
        from query_parser import QueryParser
    except ImportError as exc:
        raise RuntimeError('Place your Week 4 query_parser.py next to understand_query.py') from exc
    return QueryParser()


def understand(query,model_path='data/models/intent_classifier.joblib',threshold=0.60,parser=None):
    clf=QueryIntentClassifier.load(model_path)
    result=clf.analyze(query,uncertainty_threshold=threshold)
    parser=parser or get_week4_parser()
    result['filters']=parser.parse(query)
    result['intent_note']='Language-only inference; does not measure actual buyer behavior or conversion likelihood.'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('query')
    p.add_argument('--model',default='data/models/intent_classifier.joblib')
    p.add_argument('--threshold',type=float,default=0.60)
    args=p.parse_args()
    print(json.dumps(understand(args.query,args.model,args.threshold),indent=2))
