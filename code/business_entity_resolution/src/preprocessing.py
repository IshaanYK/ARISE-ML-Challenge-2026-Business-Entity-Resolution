import re
from anyascii import anyascii

LEGAL_STOPWORDS = {
    'inc', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'and', 'the', 'of', '&', 'sarl', 'sas', 'sasu', 'sci',
    'eurl', 'sa', 'llp', 'pllc', 'services', 'enterprises', 'solutions',
    'center', 'group', 'international', 'com', 'org', 'net', 'in', 'fr'
}

ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'blvd',
    'court', 'ct', 'dr', 'drive', 'way', 'fl', 'floor', 'suite', 'ste',
    'unit', 'apartment', 'apt', 'rue', 'allée', 'allee', 'no', 'near',
    'behind', 'opp', 'opposite', 'road', 'block', 'sector'
}

DOMAIN_RE = re.compile(r'\.(com|org|net|in|fr|co|io|biz|info)\b', re.IGNORECASE)
ALPHANUM_RE = re.compile(r'[a-z0-9]+')
DIGITS_RE = re.compile(r'\b\d+\b')


def clean_num(n: str) -> str:
    """Normalize numeric strings by removing leading zeros."""
    if not n:
        return ""
    stripped = n.lstrip('0')
    return stripped if stripped else '0'


def normalize_text(text: str) -> str:
    """Transliterate non-ASCII scripts (Indic, French, accents) to ASCII and lowercase."""
    if not text:
        return ""
    # Transliterate unicode characters to clean ASCII representations
    ascii_text = anyascii(str(text))
    # Strip domain extensions
    clean_text = DOMAIN_RE.sub('', ascii_text)
    return clean_text.lower().strip()


def soundex(name: str) -> str:
    """Standard American Soundex algorithm for phonetic key generation."""
    if not name:
        return ""
    name = name.upper()
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first_char = name[0]
    encoded = [first_char]
    prev_code = mapping.get(first_char, '')
    for char in name[1:]:
        code = mapping.get(char, '')
        if code != prev_code:
            if code:
                encoded.append(code)
            prev_code = code
    res = ''.join(encoded).replace('0', '')
    return (res + '000')[:4]


def extract_name_tokens(norm_name: str) -> list[str]:
    """Extract informative business name tokens including 2-letter initials, excluding legal stopwords."""
    if not norm_name:
        return []
    words = ALPHANUM_RE.findall(norm_name)
    tokens = [w for w in words if w not in LEGAL_STOPWORDS and len(w) >= 2]
    # Fallback if all tokens were legal stopwords: keep words except generic articles
    if not tokens:
        tokens = [w for w in words if w not in {'and', 'the', 'of', '&'}]
    return tokens


def extract_address_components(norm_addr: str) -> tuple[list[str], list[str]]:
    """Extract normalized numbers and informative text tokens from an address."""
    if not norm_addr:
        return [], []
    words = ALPHANUM_RE.findall(norm_addr)
    numbers = [clean_num(w) for w in words if w.isdigit()]
    # Keep non-empty cleaned numbers
    numbers = [n for n in numbers if n]
    
    text_words = [
        w for w in words
        if not w.isdigit() and len(w) > 2 and w not in ADDR_STOPWORDS
    ]
    return numbers, text_words


DIGIT_CHUNKS = re.compile(r'\d+')
US_STATE_NAMES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh',
    'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn',
    'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc'
}
US_STATES = set(US_STATE_NAMES.values())
sorted_state_names = sorted(US_STATE_NAMES.keys(), key=lambda s: -len(s))
FULL_STATE_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_state_names) + r')\b')
STATE_CODE_PATTERN = re.compile(r'\b[a-z]{2}\b')


def extract_all_digits(text: str) -> set[str]:
    """Extract all normalized numeric chunks from address (including embedded like 3906a)."""
    if not text:
        return set()
    return {clean_num(d) for d in DIGIT_CHUNKS.findall(text) if clean_num(d)}


def extract_state_code(norm_addr: str) -> str:
    """Extract standard 2-letter state code if present (as abbreviation or full name) in address."""
    if not norm_addr:
        return ""
    # 1. Check full state name match via fast regex
    m_full = FULL_STATE_PATTERN.findall(norm_addr)
    if m_full:
        return US_STATE_NAMES[m_full[-1]]
    # 2. Check 2-letter state code from end backwards
    words = STATE_CODE_PATTERN.findall(norm_addr)
    for w in reversed(words):
        if w in US_STATES:
            return w
    return ""


def are_numbers_compatible(nums1: set[str], nums2: set[str]) -> bool:
    """Verify numeric compatibility accounting for OCR/truncation noise without false negatives."""
    if not nums1 or not nums2:
        return True
    if nums1 & nums2:
        return True
    for n1 in nums1:
        for n2 in nums2:
            if len(n1) >= 2 and len(n2) >= 2:
                if n1.endswith(n2) or n2.endswith(n1) or n1.startswith(n2) or n2.startswith(n1):
                    return True
            if len(n1) >= 3 and len(n1) == len(n2):
                diffs = sum(c1 != c2 for c1, c2 in zip(n1, n2))
                if diffs <= 1:
                    return True
    concat1 = "".join(sorted(nums1, key=len, reverse=True))
    concat2 = "".join(sorted(nums2, key=len, reverse=True))
    for n in nums1:
        if n in concat2:
            return True
    for n in nums2:
        if n in concat1:
            return True
    return False



