import nltk
import pandas as pd
import json
from collections import Counter
from nltk.util import ngrams

df = pd.read_csv("data/processed/listing_sample.csv")

# Extract bigrams from remarks
all_text = ' '.join(df['remarks'].dropna().str.lower())
tokens = nltk.word_tokenize(all_text)
bigrams = list(ngrams(tokens, 2))
freq = Counter(bigrams)

# Top 200 bigrams become taxonomy seed
# for bigram, count in freq.most_common(200):
# 	print(f"{' '.join(bigram)}: {count}")

terms = []

for idx, (bigram, count) in enumerate(freq.most_common(200), start=1):
    terms.append({
        "id": idx,
        "term": " ".join(bigram),
        "count": count
    })

taxonomy = {
    "terms": terms
}

with open("data/processed/taxonomy.json", "w") as f:
    json.dump(taxonomy, f, indent=2)

print(f"Created taxonomy with {len(terms)} terms")