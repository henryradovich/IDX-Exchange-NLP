#!/usr/bin/env python3
"""Build and persist FAISS + BM25-ready source data from Week 2 listings."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_search import SemanticSearcher, BM25Searcher


def load_listings(csv_path):
    df = pd.read_csv(csv_path)
    text_col = "remarks_cleaned" if "remarks_cleaned" in df.columns else "remarks"
    if text_col not in df.columns:
        raise ValueError("CSV needs 'remarks_cleaned' or 'remarks'")
    id_col = "L_ListingID" if "L_ListingID" in df.columns else None
    df = df.loc[df[text_col].notna() & df[text_col].astype(str).str.strip().ne("")].copy()
    if id_col:
        df = df.drop_duplicates(subset=id_col, keep="first")
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in df.index]
    return df[text_col].astype(str).tolist(), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/listing_sample_cleaned.csv")
    ap.add_argument("--output", default="data/processed/search_index")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    remarks, ids = load_listings(args.csv)
    semantic = SemanticSearcher().build_index(remarks, ids, args.batch_size)
    semantic.save(args.output)
    print(f"Saved {len(ids)} listings; embedding dimension: {semantic.dimension}")
    bm25 = BM25Searcher().build_index(remarks, ids)
    print("BM25 baseline prepared (rebuilt from saved listings for comparison).")
    for engine in (semantic, bm25):
        print(engine.__class__.__name__, engine.search("open floor plan with mountain views", 3))

if __name__ == "__main__":
    main()
