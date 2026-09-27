import math
import polars as pl
from collections import defaultdict, Counter
from src.preprocessing import (
    normalize_text, extract_name_tokens, extract_address_components, soundex
)


def extract_blocking_keys(norm_name: str, norm_addr: str, country: str) -> list[str]:
    """Generate high-precision blocking keys from normalized name and address."""
    keys = []
    
    # 1. Distinctive name tokens, prefix n-grams, and Soundex codes
    name_words = extract_name_tokens(norm_name)
    for w in name_words:
        keys.append(f"N:{country}:{w}")
        # Prefix 4-gram for typo tolerance on longer words
        if len(w) >= 4:
            keys.append(f"NP:{country}:{w[:4]}")
        # Phonetic soundex for cross-lingual / transliterated invariance (e.g. green vs grin, shiv vs siv)
        if len(w) >= 3:
            sx = soundex(w)
            if sx:
                keys.append(f"SX:{country}:{sx}")
            
    # 2. Address numeric and text co-occurrences across multiple leading numbers
    numbers, text_words = extract_address_components(norm_addr)
    if numbers:
        for num in numbers[:2]:
            for tw in text_words[:3]:
                keys.append(f"A:{country}:{num}_{tw}")
        if len(numbers) > 1:
            keys.append(f"A:{country}:{numbers[0]}_{numbers[1]}")
    elif len(text_words) >= 2:
        keys.append(f"A:{country}:{text_words[0]}_{text_words[1]}")
            
    return keys


class CountryInvertedIndex:
    """Memory-efficient inverted index with IDF scoring for a single country."""
    def __init__(self, max_df: int = 4000):
        self.index = defaultdict(list)
        self.doc_freq = Counter()
        self.key_idf = {}
        self.target_records = {}  # id -> {entity_id, business_name, business_address, country}
        self.max_df = max_df
        self.total_docs = 0

    def build(self, df_targets: pl.DataFrame):
        """Index all target records (Source 2 and Source 3) for this country."""
        self.total_docs = len(df_targets)
        
        # Pre-normalize columns using fast columnar access
        ids = df_targets['entity_id'].to_list()
        names = df_targets['business_name'].to_list()
        addrs = df_targets['business_address'].to_list()
        countries = df_targets['country'].to_list()

        for tid, name, addr, c in zip(ids, names, addrs, countries):
            norm_n = normalize_text(name)
            norm_a = normalize_text(addr)
            
            self.target_records[tid] = {
                'entity_id': tid,
                'business_name': name,
                'business_address': addr,
                'country': c,
                'norm_name': norm_n,
                'norm_addr': norm_a
            }
            
            keys = set(extract_blocking_keys(norm_n, norm_a, c))
            for k in keys:
                self.index[k].append(tid)
                self.doc_freq[k] += 1

        # Compute IDF and prune frequent keys (df > max_df)
        pruned_keys = []
        for k, df in self.doc_freq.items():
            if df > self.max_df:
                pruned_keys.append(k)
            else:
                self.key_idf[k] = math.log(1.0 + self.total_docs / (1.0 + df))
                
        for k in pruned_keys:
            self.index.pop(k, None)

    def query_top_k(self, norm_name: str, norm_addr: str, country: str, top_k: int = 12, max_query_keys: int = 12) -> list[tuple[str, float]]:
        """Query index for top-k candidates scored by accumulated key IDF on rarest keys."""
        keys = set(extract_blocking_keys(norm_name, norm_addr, country))
        scored = [(k, self.key_idf[k]) for k in keys if k in self.key_idf]
        scored.sort(key=lambda x: -x[1])

        cand_scores = Counter()
        for k, weight in scored[:max_query_keys]:
            for cid in self.index.get(k, []):
                cand_scores[cid] += weight
                
        return cand_scores.most_common(top_k)
