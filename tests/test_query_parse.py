from filings.query_parse import parse_query


def test_detects_company_name_and_possessive():
    assert parse_query("What were Walmart's advertising costs?").ticker == "WMT"
    assert parse_query("How does JPMorgan Chase tier its models?").ticker == "JPM"
    assert parse_query("What risks does JPMorganChase face?").ticker == "JPM"  # its own spelling
    assert parse_query("When was LinkedIn fined under GDPR?").ticker == "MSFT"  # a subsidiary


def test_detects_ticker_symbol_but_not_inside_words():
    assert parse_query("What risks does NVDA list?").ticker == "NVDA"
    assert parse_query("Which pineapple suppliers matter?").ticker is None  # "apple" inside a word


def test_two_companies_means_no_filter():
    assert parse_query("Compare Apple and Microsoft revenue").ticker is None


def test_year_only_when_single_and_known():
    known = {2024, 2025}
    assert parse_query("Apple net sales in fiscal 2025?", known).year == 2025
    assert parse_query("Impact in 2024 and 2023?", known).year is None  # two years
    assert parse_query("Revenue in 2019?", known).year is None  # no 2019 filing loaded
