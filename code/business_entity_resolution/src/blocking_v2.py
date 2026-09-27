"""
Blocking V2 — High-recall inverted index.

Key improvements over V1:
  1. Character trigrams on concatenated name → catches "maurewilliamscolombier"
  2. Address keys are GUARANTEED to be included (not crowded out by name keys)
  3. Higher max_query_keys (40) → more coverage
  4. Phonetic key on full normalized name string
  5. Address-street-word keys without number → catches DBA matches with same street

Blocking recall target: >97% at K=30 (vs 92.5% in V1).
"""
import math
import re
import polars as pl
from collections import defaultdict, Counter
from src.preprocessing import (
    normalize_text, extract_name_tokens, extract_address_components, soundex,
    extract_all_digits, extract_state_code
)

ALPHANUM_RE = re.compile(r'[a-z0-9]+')


def _char_trigrams(s: str, prefix: str) -> list[str]:
    """Generate character trigrams from string (no spaces)."""
    s = re.sub(r'[^a-z0-9]', '', s)
    if len(s) < 3:
        return [f"{prefix}:{s}"] if s else []
    return [f"{prefix}:{s[i:i+3]}" for i in range(len(s) - 2)]


def _name_to_stripped(norm_name: str) -> str:
    """Concatenate informative name tokens without spaces (no legal suffixes)."""
    tokens = extract_name_tokens(norm_name)
    return ''.join(tokens)


def extract_blocking_keys_v2(norm_name: str, norm_addr: str, country: str) -> tuple[list[str], list[str]]:
    """
    Returns (name_keys, addr_keys) separately so address keys are always included.
    This prevents high-IDF name keys from crowding out address keys.
    """
    name_keys = []
    addr_keys = []
    name_words = extract_name_tokens(norm_name)
    numbers, text_words = extract_address_components(norm_addr)

    # ── NAME KEYS ──────────────────────────────────────────────────────────────
    # 1. Individual distinctive tokens + 4-char prefix + Soundex
    for w in name_words:
        name_keys.append(f"N:{country}:{w}")
        if len(w) >= 4:
            name_keys.append(f"NP:{country}:{w[:4]}")
        if len(w) >= 3:
            sx = soundex(w)
            if sx:
                name_keys.append(f"SX:{country}:{sx}")

    # 2. Word bigrams (very discriminative for multi-word names)
    if len(name_words) >= 2:
        for i in range(len(name_words) - 1):
            name_keys.append(f"NB:{country}:{name_words[i]}_{name_words[i+1]}")
        name_keys.append(f"FN:{country}:{'_'.join(name_words[:3])}")

    # 3. Character trigrams on concatenated name → catches "maurewilliamscolombier"
    stripped = _name_to_stripped(norm_name)
    if stripped:
        trigrams = _char_trigrams(stripped, f"TG:{country}")
        # Only use first 6 trigrams to limit key explosion
        name_keys.extend(trigrams[:6])

        # 4. Full-name Soundex (phonetic for transliterations like "eses phud")
        sx_full = soundex(stripped[:8].upper()) if stripped else ''
        if sx_full:
            name_keys.append(f"SXF:{country}:{sx_full}")

    # 5. Name token + Street Number super-key
    if name_words and numbers:
        name_keys.append(f"NUM_NAME:{country}:{numbers[0]}_{name_words[0]}")
        if len(name_words) > 1:
            name_keys.append(f"NUM_NAME:{country}:{numbers[0]}_{name_words[1]}")

    # 6. Name token + City/Address Word
    if name_words and text_words:
        for tw in text_words[-2:]:
            name_keys.append(f"CITY_NAME:{country}:{name_words[0]}_{tw}")

    # ── ADDRESS KEYS ──────────────────────────────────────────────────────────
    # These are kept SEPARATE so they are ALWAYS included in queries.

    # 7. Address numeric + street text co-occurrence (catches DBA/different name same address)
    if numbers:
        for num in numbers[:2]:
            for tw in text_words[:4]:
                addr_keys.append(f"A:{country}:{num}_{tw}")
        if len(numbers) > 1:
            addr_keys.append(f"A:{country}:{numbers[0]}_{numbers[1]}")
    elif len(text_words) >= 2:
        addr_keys.append(f"A:{country}:{text_words[0]}_{text_words[1]}")

    # 8. Street-name only key (no number) → catches cases where number varies slightly
    if text_words and len(text_words) >= 2:
        addr_keys.append(f"STR:{country}:{text_words[0]}_{text_words[1]}")

    # 9. Leading number + 4-char prefix of first street word (char-tolerant)
    if numbers and text_words:
        addr_keys.append(f"APC:{country}:{numbers[0]}_{text_words[0][:4]}")

    return name_keys, addr_keys


class CountryInvertedIndexV2:
    """
    High-recall inverted index with separate name/address key spaces.
    Address keys are ALWAYS included in query, never crowded out by name keys.
    """
    def __init__(self, max_df: int = 10000, max_df_trigram: int = 500):
        self.name_index = defaultdict(list)
        self.addr_index = defaultdict(list)
        self.name_idf = {}
        self.addr_idf = {}
        self.target_records = {}
        self.max_df = max_df
        self.max_df_trigram = max_df_trigram
        self.total_docs = 0

    def build(self, df_targets: pl.DataFrame):
        self.total_docs = len(df_targets)
        ids = df_targets['entity_id'].to_list()
        names = df_targets['business_name'].to_list()
        addrs = df_targets['business_address'].to_list()
        countries = df_targets['country'].to_list()

        name_df = Counter()
        addr_df = Counter()

        for tid, name, addr, c in zip(ids, names, addrs, countries):
            norm_n = normalize_text(name)
            norm_a = normalize_text(addr)

            self.target_records[tid] = {
                'entity_id': tid,
                'business_name': name,
                'business_address': addr,
                'country': c,
                'norm_name': norm_n,
                'norm_addr': norm_a,
                'nums': extract_all_digits(norm_a),
                'state': extract_state_code(norm_a),
                'words': extract_name_tokens(norm_n)
            }

            nkeys, akeys = extract_blocking_keys_v2(norm_n, norm_a, c)
            for k in set(nkeys):
                self.name_index[k].append(tid)
                name_df[k] += 1
            for k in set(akeys):
                self.addr_index[k].append(tid)
                addr_df[k] += 1

        # Compute IDF and prune high-df keys
        # Trigram keys (TG:, SXF:) use aggressive max_df to prevent bucket explosion
        for k, df in name_df.items():
            is_trigram = k.startswith('TG:') or k.startswith('SXF:')
            limit = self.max_df_trigram if is_trigram else self.max_df
            if df <= limit:
                self.name_idf[k] = math.log(1.0 + self.total_docs / (1.0 + df))
            else:
                self.name_index.pop(k, None)

        for k, df in addr_df.items():
            if df <= self.max_df:
                self.addr_idf[k] = math.log(1.0 + self.total_docs / (1.0 + df))
            else:
                self.addr_index.pop(k, None)

    def query_top_k(self, norm_name: str, norm_addr: str, country: str,
                    top_k: int = 30,
                    max_name_keys: int = 25,
                    max_addr_keys: int = 15) -> list[tuple[str, float]]:
        """
        Query with separate name/address key budgets to guarantee address coverage.
        """
        nkeys, akeys = extract_blocking_keys_v2(norm_name, norm_addr, country)

        # Score name keys by IDF, take top max_name_keys
        n_scored = [(k, self.name_idf[k]) for k in set(nkeys) if k in self.name_idf]
        n_scored.sort(key=lambda x: -x[1])

        # Score addr keys by IDF, take top max_addr_keys
        a_scored = [(k, self.addr_idf[k]) for k in set(akeys) if k in self.addr_idf]
        a_scored.sort(key=lambda x: -x[1])

        cand_scores = Counter()

        for k, weight in n_scored[:max_name_keys]:
            for cid in self.name_index.get(k, []):
                cand_scores[cid] += weight

        for k, weight in a_scored[:max_addr_keys]:
            for cid in self.addr_index.get(k, []):
                # Boost address key hits so DBA entities surface
                cand_scores[cid] += weight * 1.5

        return cand_scores.most_common(top_k)
