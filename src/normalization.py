from __future__ import annotations
import re
import unicodedata
LEGAL_ABBREV_MAP: dict[str, str] = {'pvt': 'private', 'ltd': 'limited', 'corp': 'corporation', 'inc': 'incorporated', 'co': 'company'}
LEGAL_SUFFIXES: set[str] = {'private', 'limited', 'corporation', 'incorporated', 'company', 'llc', 'llp', 'plc', 'gmbh', 'sarl', 'sa', 'sas'}
ADDRESS_ABBREV_MAP: dict[str, str] = {'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'ste': 'suite', 'apt': 'apartment', 'hwy': 'highway', 'pkwy': 'parkway', 'cir': 'circle', 'sq': 'square', 'fl': 'floor', 'bldg': 'building'}

def normalize_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ''
    name = unicodedata.normalize('NFC', name)
    name = name.lower()
    name = re.sub('\\s*&\\s*', ' and ', name)
    name = ''.join((c if unicodedata.category(c)[0] in ('L', 'M', 'N') or c.isspace() else ' ' for c in name))
    name = name.replace('_', ' ')
    name = re.sub('\\s+', ' ', name).strip()
    return name

def normalize_legal(name: str) -> str:
    if not name:
        return name
    for abbrev, full in LEGAL_ABBREV_MAP.items():
        name = re.sub(f'\\b{abbrev}\\b', full, name)
    return name

def strip_legal_suffixes(name: str) -> str:
    if not name:
        return name
    name = normalize_legal(name)
    for suffix in LEGAL_SUFFIXES:
        name = re.sub(f'\\b{re.escape(suffix)}\\b', '', name)
    name = re.sub('\\s+', ' ', name).strip()
    return name

def normalize_address(address: str) -> str:
    if not address or not isinstance(address, str):
        return ''
    address = unicodedata.normalize('NFC', address)
    address = address.lower()
    address = re.sub('\\s*&\\s*', ' and ', address)
    allowed_extras = set('-/,#')
    address = ''.join((c if unicodedata.category(c)[0] in ('L', 'M', 'N') or c.isspace() or c in allowed_extras else ' ' for c in address))
    address = address.replace('_', ' ')
    for abbrev, full in ADDRESS_ABBREV_MAP.items():
        address = re.sub(f'\\b{abbrev}\\b', full, address)
    address = re.sub('\\s+', ' ', address).strip()
    return address

def extract_postal_code(address: str) -> Optional[str]:
    if not address or not isinstance(address, str):
        return None
    m = re.search('\\b(\\d{6})\\b', address)
    if m:
        return m.group(1)
    m = re.search('\\b(\\d{5})(?:-\\d{4})?\\b', address)
    if m:
        return m.group(1)
    return None

def extract_numeric_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall('\\b\\d+\\b', text)

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return text.split()

def _sql_chain_replace(col: str, pairs: list[tuple[str, str]]) -> str:
    expr = col
    for pattern, replacement in pairs:
        expr = f"regexp_replace({expr}, '{pattern}', '{replacement}', 'g')"
    return expr

def sql_name_clean(col: str='business_name') -> str:
    steps: list[tuple[str, str]] = [('\\s*&\\s*', ' and '), ('[^\\p{L}\\p{M}\\p{N}\\s]', ' '), ('\\s+', ' ')]
    inner = _sql_chain_replace(f"lower(trim(COALESCE({col}, '')))", steps)
    return f'trim({inner})'

def sql_name_legal(clean_col: str='name_clean') -> str:
    pairs: list[tuple[str, str]] = [(f'\\b{a}\\b', f) for a, f in LEGAL_ABBREV_MAP.items()]
    return _sql_chain_replace(clean_col, pairs)

def sql_name_no_suffix(legal_col: str='name_legal') -> str:
    suffix_alt = '|'.join(LEGAL_SUFFIXES)
    stripped = f"regexp_replace({legal_col}, '\\b({suffix_alt})\\b', '', 'g')"
    return f"trim(regexp_replace({stripped}, '\\s+', ' ', 'g'))"

def sql_addr_clean(col: str='business_address') -> str:
    steps: list[tuple[str, str]] = [('\\s*&\\s*', ' and '), ('[^\\p{L}\\p{M}\\p{N}\\s\\-/,#]', ' ')]
    for abbrev, full in ADDRESS_ABBREV_MAP.items():
        steps.append((f'\\b{abbrev}\\b', full))
    steps.append(('\\s+', ' '))
    inner = _sql_chain_replace(f"lower(trim(COALESCE({col}, '')))", steps)
    return f'trim({inner})'

def sql_extract_postal(col: str='business_address') -> str:
    return f"COALESCE(regexp_extract({col}, '\\b(\\d{{6}})\\b', 1), regexp_extract({col}, '\\b(\\d{{5}})(?:-\\d{{4}})?\\b', 1))"

def build_normalization_query(tsv_path: str) -> str:
    safe_path = str(tsv_path).replace('\\', '/')
    name_clean_expr = sql_name_clean('business_name')
    addr_clean_expr = sql_addr_clean('business_address')
    postal_expr = sql_extract_postal('business_address')
    name_legal_expr = sql_name_legal('name_clean')
    name_no_suffix_expr = sql_name_no_suffix(sql_name_legal('name_clean'))
    tab = '\\t'
    return f"\n    WITH raw AS (\n        SELECT *\n        FROM read_csv('{safe_path}',\n                      delim = '{tab}',\n                      header = true,\n                      all_varchar = true,\n                      null_padding = true)\n    ),\n    base AS (\n        SELECT\n            entity_id,\n            business_name  AS raw_name,\n            business_address AS raw_address,\n            country        AS raw_country,\n            lower(trim(COALESCE(country, '')))  AS country_clean,\n            {name_clean_expr}    AS name_clean,\n            {addr_clean_expr}    AS addr_clean,\n            {postal_expr}        AS postal_code\n        FROM raw\n    )\n    SELECT\n        entity_id,\n        raw_name,\n        raw_address,\n        raw_country,\n        country_clean,\n        name_clean,\n        {name_legal_expr}        AS name_legal,\n        {name_no_suffix_expr}    AS name_no_suffix,\n        addr_clean,\n        postal_code\n    FROM base\n    "
