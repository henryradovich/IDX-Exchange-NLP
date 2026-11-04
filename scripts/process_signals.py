"""Batch extraction from CSV or the complete MySQL rets_property table."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entity_extractor import EntityExtractor
from signal_extractor import SignalExtractor


def safe_record(record):
    # Convert pandas/NumPy missing values to JSON null.
    cleaned = {}
    for k, v in record.items():
        if pd.isna(v):
            cleaned[k] = None
        elif hasattr(v, 'item'):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def rows_csv(path, chunksize):
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for row in chunk.to_dict('records'):
            yield safe_record(row)


def rows_mysql(chunksize):
    import mysql.connector
    conn = mysql.connector.connect(host=os.getenv('MYSQL_HOST', '127.0.0.1'),
        port=int(os.getenv('MYSQL_PORT', '3306')),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.environ['MYSQL_PASSWORD'],
        database=os.getenv('MYSQL_DATABASE', 'real_estate'))
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute('''SELECT L_ListingID, L_City, L_Keyword2, LM_Dec_3,
                L_SystemPrice, L_Remarks FROM rets_property''')
        while True:
            batch = cursor.fetchmany(chunksize)
            if not batch:
                break
            yield from batch
    finally:
        cursor.close()
        conn.close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--source', choices=['csv', 'mysql'], default='csv')
    cli.add_argument('--csv', default='data/processed/listing_sample_10k.csv')
    cli.add_argument('--taxonomy', default='data/processed/taxonomy.json')
    cli.add_argument('--output', default='data/processed/listing_signals.jsonl')
    cli.add_argument('--chunk-size', type=int, default=1000)
    args = cli.parse_args()
    entity = EntityExtractor(args.taxonomy)
    extractor = SignalExtractor(args.taxonomy, entity)
    source = rows_csv(args.csv, args.chunk_size) if args.source == 'csv' else rows_mysql(args.chunk_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('w', encoding='utf-8') as stream:
        for record in source:
            stream.write(json.dumps(extractor.extract_signals(record), ensure_ascii=False, default=str) + '\n')
            count += 1
    print(f'Processed {count:,} listings -> {output}')

if __name__ == '__main__':
    main()
