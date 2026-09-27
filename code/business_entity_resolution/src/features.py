import re
from rapidfuzz import fuzz
from src.preprocessing import clean_num

DIGITS_RE = re.compile(r'\b\d+\b')
WORDS_RE = re.compile(r'[a-z0-9]+')

FEATURE_NAMES = [
    # Name similarities (7)
    'n_ratio', 'n_tsort', 'n_tset', 'n_part', 'n_len_diff', 'n_len_ratio', 'n_jaccard',
    # Address presence + similarities (6)
    'addr_null', 'a_ratio', 'a_tsort', 'a_tset', 'a_part', 'a_jaccard',
    # Address numeric features (2)
    'num_common', 'first_num_match',
    # Cross-field (2)
    'cross1', 'cross2',
    # Source/blocking meta (3)
    'is_s2', 'b_score', 'b_rank',
    # Discriminative features (8)
    'n_exact_tok_overlap',
    'n_shared_bigram',
    'n_is_prefix',
    'n_abbr_match',
    'a_digit_overlap',
    'a_street_match',
    'n_wmd_approx',
    'n_s2_bonus',
]


def _char_bigrams(s: str) -> set:
    return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def _acronym(tokens: list) -> str:
    return ''.join(t[0] for t in tokens if t)


def extract_pair_features(
    norm_n1: str, norm_a1: str,
    norm_n2: str, norm_a2: str,
    target_id: str,
    b_score: float, b_rank: int
) -> list[float]:
    """Compute rich 28-dimensional similarity vector between S1 record and candidate target."""
    # ── Name similarities ──────────────────────────────────────────────────
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

    # ── Address similarities ───────────────────────────────────────────────
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

    # ── Cross-field similarities ───────────────────────────────────────────
    cross1 = fuzz.partial_ratio(norm_n1, norm_a2) if norm_a2 else 0.0
    cross2 = fuzz.partial_ratio(norm_n2, norm_a1) if norm_a1 else 0.0

    is_s2 = 1.0 if target_id.startswith('S2-') else 0.0

    # ── Discriminative features ────────────────────────────────────────────
    # 1. Exact token overlap ratio
    n_exact_tok_overlap = float(len(w1 & w2)) / float(max(len(w1), len(w2), 1))

    # 2. Character bigram overlap
    bg1 = _char_bigrams(norm_n1.replace(' ', ''))
    bg2 = _char_bigrams(norm_n2.replace(' ', ''))
    n_shared_bigram = float(len(bg1 & bg2)) / float(len(bg1 | bg2)) if (bg1 or bg2) else 0.0

    # 3. Prefix containment
    shorter, longer = (norm_n1, norm_n2) if len1 <= len2 else (norm_n2, norm_n1)
    n_is_prefix = 1.0 if (shorter and longer.startswith(shorter)) else 0.0

    # 4. Acronym match
    acr1 = _acronym(list(w1))
    acr2 = _acronym(list(w2))
    n_abbr_match = 1.0 if (acr1 and acr2 and (
        acr1 == norm_n2.replace(' ', '') or
        acr2 == norm_n1.replace(' ', '') or
        acr1 == acr2
    )) else 0.0

    # 5. Address digit set overlap
    a_digit_overlap = 0.0
    if norm_a2:
        digs1 = set(DIGITS_RE.findall(norm_a1))
        digs2 = set(DIGITS_RE.findall(norm_a2))
        if digs1 or digs2:
            a_digit_overlap = float(len(digs1 & digs2)) / float(len(digs1 | digs2))

    # 6. First non-numeric address token match (fast: first token only, no loop)
    a_street_match = 0.0
    if norm_a2:
        st1 = [t for t in WORDS_RE.findall(norm_a1) if not t.isdigit() and len(t) > 3]
        st2 = [t for t in WORDS_RE.findall(norm_a2) if not t.isdigit() and len(t) > 3]
        if st1 and st2:
            a_street_match = 1.0 if st1[0] == st2[0] else float(fuzz.ratio(st1[0], st2[0])) / 100.0

    # 7. Word-mover proxy — FAST: cap at 2 tokens max to avoid O(n^2) slowdown
    n_wmd_approx = 0.0
    if w1 and w2:
        s_small = list(w1)[:2]   # hard cap at 2 tokens
        s_large = list(w2)[:2]
        scores = [max((fuzz.ratio(tok, t) for t in s_large), default=0) for tok in s_small]
        n_wmd_approx = float(sum(scores)) / (len(scores) * 100.0) if scores else 0.0

    # 8. S2 source bonus
    n_s2_bonus = 1.0 if target_id.startswith('S2-') else 0.0

    return [
        n_ratio, n_tsort, n_tset, n_part, n_len_diff, n_len_ratio, n_jaccard,
        addr_null, a_ratio, a_tsort, a_tset, a_part, a_jaccard,
        num_common, first_num_match,
        cross1, cross2, is_s2, float(b_score), float(b_rank),
        n_exact_tok_overlap, n_shared_bigram, n_is_prefix, n_abbr_match,
        a_digit_overlap, a_street_match, n_wmd_approx, n_s2_bonus,
    ]
