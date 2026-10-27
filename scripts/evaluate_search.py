#!/usr/bin/env python3
"""Week 5: benchmark FAISS + BM25 on 10,000 distinct real listings.

Run from the project root. Imports semantic_search.py and build_search_index.py
from this script's directory; no scripts/__init__.py needed.

The index is rebuilt from the chosen CSV, eliminating stale-1k-index errors.
Relevance judgments are optional and reported separately from latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_search import BM25Searcher, SemanticSearcher
from build_search_index import load_listings

DEFAULT_QUERIES = [
    "three bedroom home with hardwood floors",
    "modern home with mountain views",
    "granite countertops and stainless steel appliances",
    "house with a swimming pool and patio",
    "open floor plan with natural light",
    "quiet neighborhood near parks",
    "four bedroom home under 700k",
    "solar panels and an attached garage",
    "updated kitchen and primary suite",
    "new construction with a home office",
]


def stats_ms(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(x.mean()), 3),
        "p50": round(float(np.percentile(x, 50)), 3),
        "p95": round(float(np.percentile(x, 95)), 3),
        "p99": round(float(np.percentile(x, 99)), 3),
        "max": round(float(x.max()), 3),
        "runs": int(x.size),
    }


def measure(engine, queries, top_k, warmup, runs):
    for i in range(warmup):
        engine.search(queries[i % len(queries)], top_k)
    elapsed = []
    for i in range(runs):
        start = time.perf_counter()
        result = engine.search(queries[i % len(queries)], top_k)
        elapsed.append((time.perf_counter() - start) * 1000)
        if len(result) != top_k:
            raise RuntimeError(f"Expected {top_k} results, got {len(result)}")
    return stats_ms(elapsed)


def load_judgments(path):
    grouped = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("relevance") or "").strip()
            if not label:
                continue
            value = int(label)
            if value not in (0, 1, 2):
                raise ValueError("Relevance labels must be 0, 1, or 2")
            grouped[row["query"]][str(row["listing_id"])] = value
    return grouped


def evaluate_judged_at_k(engine, judgments, top_k, corpus_ids):
    """Judged-only descriptive metrics; not unbiased full-corpus retrieval metrics.

    Unjudged retrieved results are NOT assumed irrelevant. If not all top-k
    results are judged, return coverage statistics instead of P@k/NDCG@k.
    """
    rows = []
    for query, judged in judgments.items():
        judged = {ident: grade for ident, grade in judged.items() if ident in corpus_ids}
        if not judged:
            continue
        retrieved = engine.search(query, top_k)
        inspected = sum(r["id"] in judged for r in retrieved)
        rows.append({"query": query, "judged_in_top_k": inspected, "top_k": len(retrieved)})
    return {
        "queries_with_in_corpus_judgments": len(rows),
        "mean_judged_fraction_of_top_k": round(statistics.mean(r["judged_in_top_k"] / r["top_k"] for r in rows), 4) if rows else None,
        "note": "Top-k precision/NDCG withheld unless retrieved results are comprehensively judged; unknown relevance is not zero.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/processed/listing_sample_10k.csv")
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--runs", "--latency-runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--report", default="reports/search_comparison_10k.json")
    ap.add_argument("--save-index", default=None, help="Optional directory to persist FAISS index")
    ap.add_argument("--judgments", default=None, help="Optional labeled CSV for coverage diagnostic")
    args = ap.parse_args()

    if args.runs < 1 or args.warmup < 0 or args.target < 1 or args.top_k < 1 or args.top_k > args.target:
        ap.error("Require runs >= 1, warmup >= 0, 1 <= top-k <= target")

    print(f"Loading {args.csv} ...", flush=True)
    remarks, ids = load_listings(args.csv)
    if len(ids) < args.target:
        raise ValueError(f"Need {args.target:,} distinct nonempty listings, found {len(ids):,}. Check input CSV.")
    remarks, ids = remarks[:args.target], ids[:args.target]
    if len(set(ids)) != args.target:
        raise ValueError("Listing IDs must be unique")

    t0 = time.perf_counter()
    semantic = SemanticSearcher()
    model_load_s = time.perf_counter() - t0

    print(f"Building embeddings and FAISS index for {len(ids):,} real listings...", flush=True)
    t0 = time.perf_counter()
    semantic.build_index(remarks, ids, batch_size=args.batch_size)
    faiss_build_s = time.perf_counter() - t0
    if semantic.index.ntotal != args.target:
        raise RuntimeError("FAISS index size mismatch")
    if args.save_index:
        semantic.save(args.save_index)

    print("Building BM25 baseline...", flush=True)
    t0 = time.perf_counter()
    bm25 = BM25Searcher().build_index(remarks, ids)
    bm25_build_s = time.perf_counter() - t0

    print(f"Running {args.warmup} warmups and {args.runs} timed queries for each engine...", flush=True)
    latency = {
        "semantic": measure(semantic, DEFAULT_QUERIES, args.top_k, args.warmup, args.runs),
        "bm25": measure(bm25, DEFAULT_QUERIES, args.top_k, args.warmup, args.runs),
    }
    report = {
        "listing_count": args.target,
        "data_source": str(args.csv),
        "corpus_type": "real_distinct_listing_ids",
        "embedding_dimension": int(semantic.dimension),
        "top_k": args.top_k,
        "warmup_runs_per_engine": args.warmup,
        "timed_queries_per_engine": args.runs,
        "index_build_seconds": {
            "model_load": round(model_load_s, 3),
            "semantic_embeddings_and_faiss": round(faiss_build_s, 3),
            "bm25": round(bm25_build_s, 3),
        },
        "latency_ms": latency,
        "latency_scope": "warm end-to-end engine.search: query embedding/tokenization, vector or keyword lookup, and result construction; excludes indexing/model loading and CSV I/O",
        "threshold_ms": 100,
        "pass_mean_below_100ms": latency["semantic"]["mean"] < 100,
        "pass_p95_below_100ms": latency["semantic"]["p95"] < 100,
    }
    if args.judgments:
        judged = load_judgments(args.judgments)
        corpus_ids = set(ids)
        report["relevance_judgment_coverage"] = {
            name: evaluate_judged_at_k(engine, judged, args.top_k, corpus_ids)
            for name, engine in (("semantic", semantic), ("bm25", bm25))
        }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()