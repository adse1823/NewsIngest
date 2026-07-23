import importlib.util
import sys
from pathlib import Path

import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_build_graph():
    name = "graph.build_graph"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "graph" / "build_graph.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BG = _load_build_graph()

TICKER_IDX = {t: i for i, t in enumerate(
    ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
)}


# ── tests: build_cooccurrence ─────────────────────────────────────────────────

def test_single_company_mention_produces_no_edges():
    # Avoid "earnings" — "gs" (GS keyword) is a substring of that word
    titles = [["Apple stock rises today"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    assert len(co) == 0


def test_two_companies_in_one_title_creates_one_edge():
    titles = [["Apple and Microsoft partner on cloud"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    assert len(co) == 1


def test_three_companies_creates_three_edges():
    titles = [["Apple Microsoft Google cloud deal"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    assert len(co) == 3


def test_no_self_loops_in_edges():
    titles = [["Apple and Microsoft and Tesla deal"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    for (a, b) in co.keys():
        assert a != b


def test_edge_keys_are_canonical_min_first():
    titles = [["Apple and Microsoft announce deal"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    for (a, b) in co.keys():
        assert a < b


def test_repeated_pair_increments_weight():
    titles = [
        ["Apple and Microsoft cloud"],
        ["Apple and Microsoft cloud"],
    ]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    aapl_i = TICKER_IDX["AAPL"]
    msft_i = TICKER_IDX["MSFT"]
    key = (min(aapl_i, msft_i), max(aapl_i, msft_i))
    # The double-loop adds 2 per window per pair (once for (a,b), once for (b,a))
    # 2 windows × 2 = 4
    assert co[key] == 4


def test_empty_window_list_returns_empty_dict():
    co = BG.build_cooccurrence([], TICKER_IDX)
    assert co == {}


def test_empty_titles_within_window_returns_empty_dict():
    co = BG.build_cooccurrence([[]], TICKER_IDX)
    assert co == {}


def test_matching_is_case_insensitive():
    # "earnings" avoided — "gs" (GS keyword) is a substring of that word
    titles = [["APPLE and MICROSOFT announce"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    assert len(co) == 1


def test_keyword_aliases_are_used():
    # "alphabet" maps to GOOGL, "tesla" maps to TSLA
    titles = [["Alphabet and Tesla announce AI partnership"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    assert len(co) == 1


def test_unrelated_text_produces_no_edges():
    titles = [["The weather is nice today"]]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    assert len(co) == 0


def test_multiple_windows_accumulate_weights():
    titles = [
        ["Apple and Microsoft"],
        ["Apple and Microsoft"],
        ["Apple and Microsoft"],
    ]
    co = BG.build_cooccurrence(titles, TICKER_IDX)
    aapl_i = TICKER_IDX["AAPL"]
    msft_i = TICKER_IDX["MSFT"]
    key = (min(aapl_i, msft_i), max(aapl_i, msft_i))
    # 3 windows × 2 (double-loop per pair) = 6
    assert co[key] == 6
