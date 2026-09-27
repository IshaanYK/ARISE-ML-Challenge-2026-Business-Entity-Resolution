import re
from rapidfuzz import fuzz
from src.preprocessing import clean_num

DIGITS_RE = re.compile(r'\b\d+\b')
WORDS_RE = re.compile(r'[a-z0-9]+')

FEATURE_NAMES = [
    'n_ratio', 'n_tsort', 'n_tset', 'n_part', 'n_len_diff', 'n_len_ratio',
    'n_jaccard', 'addr_null', 'a_ratio', 'a_tsort', 'a_tset', 'a_part',
    'a_jaccard', 'num_common', 'first_num_match',
    'cross1', 'cross2', 'is_s2', 'b_score', 'b_rank'
]


def extract_pair_features(
    norm_n1: str, norm_a1: str,
    norm_n2: str, norm_a2: str,
    target_id: str,
    b_score: float, b_rank: int
) -> list[float]:
    """Compute rich similarity vector between Source 1 record and candidate target record."""
    # Name similarities
    n_ratio = fuzz.ratio(norm_n1, norm_n2)
    n_tsort = fuzz.token_sort_ratio(norm_n1, norm_n2)
    n_tset = fuzz.token_set_ratio(norm_n1, norm_n2)
    n_part = fuzz.partial_ratio(norm_n1, norm_n2)
    
    len1, len2 = len(norm_n1), len(norm_n2)
    n_len_diff = float(abs(len1 - len2))
    n_len_ratio = float(min(len1, len2)) / float(max(len1, len2, 1))
    
    w1 = set(WORDS_RE.findall(norm_n1))
    w2 = set(WORDS_RE.findall(norm_n2))
    n_jaccard = float(len(w1 & w2)) / float(len(w1 | w2)) if (w1 or w2) else 0.0

    # Address similarities
    addr_null = 1.0 if not norm_a2 else 0.0
    if norm_a2:
        a_ratio = fuzz.ratio(norm_a1, norm_a2)
        a_tsort = fuzz.token_sort_ratio(norm_a1, norm_a2)
        a_tset = fuzz.token_set_ratio(norm_a1, norm_a2)
        a_part = fuzz.partial_ratio(norm_a1, norm_a2)
        
        aw1 = set(WORDS_RE.findall(norm_a1))
        aw2 = set(WORDS_RE.findall(norm_a2))
        a_jaccard = float(len(aw1 & aw2)) / float(len(aw1 | aw2)) if (aw1 or aw2) else 0.0
        
        nums1 = {clean_num(n) for n in DIGITS_RE.findall(norm_a1)}
        nums2 = {clean_num(n) for n in DIGITS_RE.findall(norm_a2)}
        num_common = float(len(nums1 & nums2))
        
        m1 = DIGITS_RE.search(norm_a1)
        m2 = DIGITS_RE.search(norm_a2)
        first_num_match = 1.0 if (m1 and m2 and clean_num(m1.group()) == clean_num(m2.group())) else 0.0
    else:
        a_ratio = a_tsort = a_tset = a_part = a_jaccard = num_common = first_num_match = 0.0

    # Cross-field similarities
    cross1 = fuzz.partial_ratio(norm_n1, norm_a2) if norm_a2 else 0.0
    cross2 = fuzz.partial_ratio(norm_n2, norm_a1) if norm_a1 else 0.0
    
    is_s2 = 1.0 if target_id.startswith('S2-') else 0.0

    return [
        n_ratio, n_tsort, n_tset, n_part, n_len_diff, n_len_ratio,
        n_jaccard, addr_null, a_ratio, a_tsort, a_tset, a_part,
        a_jaccard, num_common, first_num_match,
        cross1, cross2, is_s2, float(b_score), float(b_rank)
    ]
