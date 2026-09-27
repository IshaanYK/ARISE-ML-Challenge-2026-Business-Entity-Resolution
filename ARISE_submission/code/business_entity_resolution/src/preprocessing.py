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

