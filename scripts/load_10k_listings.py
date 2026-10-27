
import os
from pathlib import Path

import mysql.connector
import pandas as pd

OUTPUT_PATH = Path("data/processed/listing_sample_10k.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.getenv("MYSQL_PASSWORD", "root"),
    database="real_estate",
)

query = """
SELECT
    L_ListingID,
    L_Address,
    L_City,
    L_Keyword2 AS beds,
    LM_Dec_3 AS baths,
    L_SystemPrice AS price,
    L_Remarks AS remarks
FROM rets_property
WHERE L_Remarks IS NOT NULL
  AND LENGTH(L_Remarks) > 50
  AND L_ListingID IS NOT NULL
GROUP BY
    L_ListingID,
    L_Address,
    L_City,
    L_Keyword2,
    LM_Dec_3,
    L_SystemPrice,
    L_Remarks
ORDER BY RAND()
LIMIT 10000
"""

try:
    cursor = conn.cursor()
    cursor.execute(query)

    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)

    # Ensure one row per listing ID
    df = df.drop_duplicates(subset=["L_ListingID"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Extracted {len(df):,} unique listings")
    print(f"Saved to: {OUTPUT_PATH}")

    if len(df) < 10000:
        print(
            "WARNING: Fewer than 10,000 unique listings were extracted."
        )

finally:
    conn.close()
