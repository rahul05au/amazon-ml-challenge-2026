"""Business name and address normalization for entity resolution.

Provides two parallel implementations:

1. **Pure-Python functions** — used by unit tests and small-scale processing.
2. **DuckDB SQL builders** — generate the same logic as composable SQL
   expressions for large-scale (millions of rows) processing inside DuckDB.

Design notes
------------
* Conservative normalization — tokens are lowered and cleaned, but
  *not* removed just because they look common.  Tokens like "india"
  or "services" may carry identity information.
* Three name variants are produced:
    ``name_clean``        – case/punct/whitespace normalised, original tokens kept
    ``name_legal``        – legal-suffix abbreviations expanded (pvt→private …)
    ``name_no_suffix``    – legal suffixes stripped entirely
* Unicode is NFC-normalised so that Devanagari and Latin text compare
  correctly.
"""

from __future__ import annotations

import re
import unicodedata

# Abbreviation → canonical full form.
LEGAL_ABBREV_MAP: dict[str, str] = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
}

# Full-form suffixes to strip (after abbreviation expansion).
LEGAL_SUFFIXES: set[str] = {
    "private", "limited", "corporation", "incorporated",
    "company", "llc", "llp", "plc", "gmbh", "sarl", "sa", "sas",
}

ADDRESS_ABBREV_MAP: dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "ste": "suite",
    "apt": "apartment",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "sq": "square",
    "fl": "floor",
    "bldg": "building",
}


def normalize_name(name: str) -> str:
    """Conservative name normalisation.

    Applies: NFC → lowercase → ``&`` → ``and`` → strip punctuation
    (keeping Unicode letters/digits) → collapse whitespace.
    """
    if not name or not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFC", name)
    name = name.lower()
    name = re.sub(r"\s*&\s*", " and ", name)
    # Remove anything that is not a Unicode letter, combining mark, digit, or whitespace.
    name = "".join(
        c if (unicodedata.category(c)[0] in ("L", "M", "N") or c.isspace()) else " "
        for c in name
    )
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_legal(name: str) -> str:
    """Expand legal-suffix abbreviations in an already-cleaned name.

    ``pvt`` → ``private``, ``ltd`` → ``limited``, etc.
    """
    if not name:
        return name
    for abbrev, full in LEGAL_ABBREV_MAP.items():
        name = re.sub(rf"\b{abbrev}\b", full, name)
    return name


def strip_legal_suffixes(name: str) -> str:
    """Remove all legal suffixes from an already-cleaned name.

    Expands abbreviations first, then strips canonical forms.
    """
    if not name:
        return name
    name = normalize_legal(name)
    for suffix in LEGAL_SUFFIXES:
        name = re.sub(rf"\b{re.escape(suffix)}\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_address(address: str) -> str:
    """Conservative address normalisation.

    Keeps hyphens (postal codes) and slashes (unit numbers).
    """
    if not address or not isinstance(address, str):
        return ""
    address = unicodedata.normalize("NFC", address)
    address = address.lower()
    address = re.sub(r"\s*&\s*", " and ", address)
    # Keep letters, combining marks, digits, whitespace, hyphens, slashes, commas, hash
    allowed_extras = set("-/,#")
    address = "".join(
        c if (unicodedata.category(c)[0] in ("L", "M", "N") or c.isspace() or c in allowed_extras) else " "
        for c in address
    )
    address = address.replace("_", " ")
    for abbrev, full in ADDRESS_ABBREV_MAP.items():
        address = re.sub(rf"\b{abbrev}\b", full, address)
    address = re.sub(r"\s+", " ", address).strip()
    return address


def extract_postal_code(address: str) -> Optional[str]:
    """Extract postal / PIN / ZIP code from raw address text.

    Checks 6-digit (India PIN) first, then 5-digit (US / France ZIP).
    Returns ``None`` when no plausible code is found.
    """
    if not address or not isinstance(address, str):
        return None
    m = re.search(r"\b(\d{6})\b", address)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    if m:
        return m.group(1)
    return None


def extract_numeric_tokens(text: str) -> list[str]:
    """Return all standalone numeric tokens (house numbers, units …)."""
    if not text:
        return []
    return re.findall(r"\b\d+\b", text)


def tokenize(text: str) -> list[str]:
    """Split cleaned text into whitespace-separated tokens."""
    if not text:
        return []
    return text.split()


# Each function returns a SQL *expression* (not a full statement) that can
# be embedded in a SELECT list.  The input is the name of a column.
#
# Escaping contract:
#   DuckDB single-quoted strings do NOT interpret backslashes as escapes.
#   So the Python string  r"\s+"  when placed in SQL  '\s+'  means RE2
#   sees the pattern  \s+  — which is correct.
#   Use plain backslash sequences everywhere in the pattern strings.

def _sql_chain_replace(col: str, pairs: list[tuple[str, str]]) -> str:
    """Nest ``regexp_replace`` calls for a list of (pattern, replacement)."""
    expr = col
    for pattern, replacement in pairs:
        expr = f"regexp_replace({expr}, '{pattern}', '{replacement}', 'g')"
    return expr


def sql_name_clean(col: str = "business_name") -> str:
    r"""SQL expression: conservative name normalisation."""
    steps: list[tuple[str, str]] = [
        # &→and (before punctuation strip so & is gone)
        (r"\s*&\s*", " and "),
        # Remove non-letter/digit/space. RE2 Unicode property classes (including \p{M} combining marks).
        (r"[^\p{L}\p{M}\p{N}\s]", " "),
        (r"\s+", " "),
    ]
    inner = _sql_chain_replace(f"lower(trim(COALESCE({col}, '')))", steps)
    return f"trim({inner})"


def sql_name_legal(clean_col: str = "name_clean") -> str:
    r"""SQL expression: expand legal abbreviations on an already-clean name."""
    pairs: list[tuple[str, str]] = [
        (rf"\b{a}\b", f) for a, f in LEGAL_ABBREV_MAP.items()
    ]
    return _sql_chain_replace(clean_col, pairs)


def sql_name_no_suffix(legal_col: str = "name_legal") -> str:
    r"""SQL expression: strip legal suffixes from legal-normalised name."""
    suffix_alt = "|".join(LEGAL_SUFFIXES)
    stripped = rf"regexp_replace({legal_col}, '\b({suffix_alt})\b', '', 'g')"
    return rf"trim(regexp_replace({stripped}, '\s+', ' ', 'g'))"


def sql_addr_clean(col: str = "business_address") -> str:
    r"""SQL expression: conservative address normalisation."""
    steps: list[tuple[str, str]] = [
        (r"\s*&\s*", " and "),
        # Keep letters, combining marks, digits, whitespace, hyphen, slash, comma, hash
        (r"[^\p{L}\p{M}\p{N}\s\-/,#]", " "),
    ]
    for abbrev, full in ADDRESS_ABBREV_MAP.items():
        steps.append((rf"\b{abbrev}\b", full))
    steps.append((r"\s+", " "))
    inner = _sql_chain_replace(f"lower(trim(COALESCE({col}, '')))", steps)
    return f"trim({inner})"


def sql_extract_postal(col: str = "business_address") -> str:
    r"""SQL expression: extract postal/PIN code (prefers 6-digit, then 5-digit)."""
    return (
        f"COALESCE("
        rf"regexp_extract({col}, '\b(\d{{6}})\b', 1), "
        rf"regexp_extract({col}, '\b(\d{{5}})(?:-\d{{4}})?\b', 1)"
        f")"
    )


def build_normalization_query(tsv_path: str) -> str:
    """Return a complete DuckDB SQL query that reads a TSV and produces
    all normalised columns.

    The result can be used directly as:
        ``con.execute(f"CREATE TABLE norm AS {query}")``
    or written to Parquet.
    """
    # Escape backslashes in Windows paths for SQL
    safe_path = str(tsv_path).replace("\\", "/")

    name_clean_expr = sql_name_clean("business_name")
    addr_clean_expr = sql_addr_clean("business_address")
    postal_expr = sql_extract_postal("business_address")
    name_legal_expr = sql_name_legal("name_clean")
    name_no_suffix_expr = sql_name_no_suffix(sql_name_legal("name_clean"))

    # Use explicit tab delimiter string to avoid Python \t interpretation
    tab = "\\t"

    return f"""
    WITH raw AS (
        SELECT *
        FROM read_csv('{safe_path}',
                      delim = '{tab}',
                      header = true,
                      all_varchar = true,
                      null_padding = true)
    ),
    base AS (
        SELECT
            entity_id,
            business_name  AS raw_name,
            business_address AS raw_address,
            country        AS raw_country,
            lower(trim(COALESCE(country, '')))  AS country_clean,
            {name_clean_expr}    AS name_clean,
            {addr_clean_expr}    AS addr_clean,
            {postal_expr}        AS postal_code
        FROM raw
    )
    SELECT
        entity_id,
        raw_name,
        raw_address,
        raw_country,
        country_clean,
        name_clean,
        {name_legal_expr}        AS name_legal,
        {name_no_suffix_expr}    AS name_no_suffix,
        addr_clean,
        postal_code
    FROM base
    """

