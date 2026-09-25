"""Unit tests for normalization functions.

Run with:  python -m pytest tests/test_normalization.py -v
"""

import pytest
from src.normalization import (
    normalize_name,
    normalize_legal,
    strip_legal_suffixes,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    tokenize,
)


# ── normalize_name ─────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_basic(self):
        assert normalize_name("ABC Corp.") == "abc corp"

    def test_ampersand(self):
        assert normalize_name("Smith & Sons") == "smith and sons"

    def test_multiple_ampersands(self):
        assert normalize_name("A & B & C") == "a and b and c"

    def test_whitespace_collapse(self):
        assert normalize_name("  Multiple   Spaces  ") == "multiple spaces"

    def test_punctuation(self):
        assert normalize_name("Hello, World! (test)") == "hello world test"

    def test_empty(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_hindi_devanagari(self):
        """Devanagari characters must survive normalisation."""
        result = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
        assert "राम" in result
        assert "मार्केटिंग" in result

    def test_mixed_script(self):
        result = normalize_name("ABC — राम Pvt. Ltd.")
        assert "abc" in result
        assert "राम" in result
        assert "pvt" in result

    def test_preserves_digits(self):
        assert normalize_name("Unit 42B") == "unit 42b"


# ── normalize_legal ────────────────────────────────────────────────────────

class TestNormalizeLegal:
    def test_pvt_to_private(self):
        assert normalize_legal("xyz pvt") == "xyz private"

    def test_ltd_to_limited(self):
        assert normalize_legal("abc ltd") == "abc limited"

    def test_corp_to_corporation(self):
        assert normalize_legal("mega corp") == "mega corporation"

    def test_inc_to_incorporated(self):
        assert normalize_legal("acme inc") == "acme incorporated"

    def test_co_to_company(self):
        assert normalize_legal("smith and co") == "smith and company"

    def test_combined(self):
        assert normalize_legal("xyz pvt ltd") == "xyz private limited"

    def test_no_partial_match(self):
        """'co' inside 'costco' must not be replaced."""
        assert normalize_legal("costco") == "costco"


# ── strip_legal_suffixes ──────────────────────────────────────────────────

class TestStripLegalSuffixes:
    def test_private_limited(self):
        assert strip_legal_suffixes("abc private limited") == "abc"

    def test_pvt_ltd_abbreviation(self):
        """Abbreviations should be expanded first, then stripped."""
        assert strip_legal_suffixes("abc pvt ltd") == "abc"

    def test_llc(self):
        assert strip_legal_suffixes("acme llc") == "acme"

    def test_nothing_to_strip(self):
        assert strip_legal_suffixes("acme bakery") == "acme bakery"

    def test_empty(self):
        assert strip_legal_suffixes("") == ""


# ── normalize_address ─────────────────────────────────────────────────────

class TestNormalizeAddress:
    def test_basic(self):
        result = normalize_address("123 Main St.")
        assert "123 main street" in result

    def test_road(self):
        assert "road" in normalize_address("456 Oak Rd")

    def test_avenue(self):
        assert "avenue" in normalize_address("789 Park Ave")

    def test_boulevard(self):
        assert "boulevard" in normalize_address("10 Sunset Blvd")

    def test_keeps_hyphens(self):
        """Postal codes like 28601 or 110001 may appear with hyphens."""
        result = normalize_address("NC 28601-1234")
        assert "28601-1234" in result

    def test_keeps_slashes(self):
        result = normalize_address("KH NO. 570/13")
        assert "570/13" in result

    def test_empty(self):
        assert normalize_address("") == ""
        assert normalize_address(None) == ""

    def test_hindi_address(self):
        result = normalize_address("G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh")
        assert "bhopal" in result
        assert "madhya pradesh" in result


# ── extract_postal_code ───────────────────────────────────────────────────

class TestExtractPostalCode:
    def test_india_pin(self):
        assert extract_postal_code("NEW DELHI, 110001, Delhi") == "110001"

    def test_us_zip5(self):
        assert extract_postal_code("Morganton, NC 28601") == "28601"

    def test_us_zip_plus4(self):
        assert extract_postal_code("PO Box 6009, Cincinnati, OH 45206-1234") == "45206"

    def test_france_postal(self):
        assert extract_postal_code("75001 Paris, France") == "75001"

    def test_no_code(self):
        assert extract_postal_code("Some Address Without Code") is None

    def test_empty(self):
        assert extract_postal_code("") is None
        assert extract_postal_code(None) is None

    def test_six_digit_preferred_over_five(self):
        """A 6-digit code (Indian PIN) should be preferred over a 5-digit one
        that might be embedded inside it."""
        assert extract_postal_code("PIN 400001") == "400001"


# ── extract_numeric_tokens ────────────────────────────────────────────────

class TestExtractNumericTokens:
    def test_basic(self):
        assert extract_numeric_tokens("Unit 42, Floor 3") == ["42", "3"]

    def test_none(self):
        assert extract_numeric_tokens("no digits here") == []

    def test_empty(self):
        assert extract_numeric_tokens("") == []


# ── tokenize ──────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize(None) == []
