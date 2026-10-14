"""Tests for numbers.py: extracting numbers from filing text and matching them."""

from filings.numbers import extract_numbers, numbers_match, same_number, unsupported_numbers


def test_extract_skips_years_and_small_counts():
    text = "In fiscal 2025, net sales were $391,035 million across 2 segments, up 2.1%."
    assert extract_numbers(text) == [391035.0, 2.1]


def test_extract_reads_accounting_negatives():
    assert extract_numbers("Derivatives CVA | (14) | 25") == [-14.0, 25.0]


def test_same_number_allows_rounding_and_unit_changes():
    assert same_number(11.5, 11504)  # $11.5 billion vs. 11,504 (in millions)
    assert same_number(5.1, 5.1)
    assert not same_number(5.1, 4.4)


def test_numbers_match():
    gold = "Advertising costs were $5.1 billion and $4.4 billion."
    assert numbers_match(gold, "They were $5.1B in 2025 and $4.4B in 2024.") is True
    assert numbers_match(gold, "They were $5.1B in 2025 and $4.1B in 2024.", all_numbers=True) is False
    assert numbers_match(gold, "They were $5.1B in 2025.") is True  # main figure is enough
    assert numbers_match(gold, "They were $5.4B in 2025.") is False
    assert numbers_match("Apple has a minority market share.", "anything") is None


def test_unsupported_numbers():
    sources = ["Total cash taxes paid, net of refunds | 11,179 | 12,308"]
    assert unsupported_numbers("Amazon paid $11.2 billion.", sources) == []
    assert unsupported_numbers("Amazon paid $13.9 billion.", sources) == [13.9]


def test_extract_skips_days_in_dates():
    assert extract_numbers("As of December 31, 2024, debt was $10,883 million.") == [10883.0]


def test_same_number_thousands_vs_billions():
    assert same_number(14.773, 14_773_260)  # 14.773 billion shares vs. 14,773,260 thousand


def test_extract_skips_references():
    assert extract_numbers("As reported in Note 19, liabilities were $35,106 million.") == [35106.0]
