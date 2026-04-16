import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.

    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)

    Returns:
        requests.Response: The response object

    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue

        # Return the response (whether success, other errors, or final 429)
        return response


def _get_headers(api_key: str = None) -> dict:
    """Build auth headers for FD API."""
    headers = {}
    key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if key:
        headers["X-API-KEY"] = key
    return headers


# ---------------------------------------------------------------------------
# Prices (free tier — always fetch, basic same-run cache)
# ---------------------------------------------------------------------------
def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    headers = _get_headers(api_key)
    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        return []

    try:
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
    except Exception as e:
        logger.warning("Failed to parse price response for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


# ---------------------------------------------------------------------------
# Financial metrics — PROBE-BASED smart cache
#   1. Check if we have cached data with a known latest_report_period
#   2. Probe API with limit=1 to get the newest report_period
#   3. If same → return cached data (no extra cost)
#   4. If different → full fetch and update cache
# ---------------------------------------------------------------------------
def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics with probe-based smart caching."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    headers = _get_headers(api_key)

    # Check if we have cached data
    cached_data = _cache.get_financial_metrics(cache_key)
    cached_report_period = _cache.get_latest_report_period("financial_metrics", cache_key)

    if cached_data and cached_report_period:
        # PROBE: fetch limit=1 to see if latest report_period changed
        probe_url = (
            f"https://api.financialdatasets.ai/financial-metrics/"
            f"?ticker={ticker}&report_period_lte={end_date}&limit=1&period={period}"
        )
        try:
            probe_resp = _make_api_request(probe_url, headers)
            if probe_resp.status_code == 200:
                probe_data = probe_resp.json()
                probe_metrics = probe_data.get("financial_metrics", [])
                if probe_metrics:
                    latest_rp = probe_metrics[0].get("report_period", "")
                    if latest_rp == cached_report_period:
                        logger.info(
                            "Smart cache HIT for financial_metrics %s "
                            "(report_period unchanged: %s)",
                            ticker, latest_rp,
                        )
                        return [FinancialMetrics(**m) for m in cached_data]
                    else:
                        logger.info(
                            "Smart cache MISS for financial_metrics %s "
                            "(new report_period: %s → %s)",
                            ticker, cached_report_period, latest_rp,
                        )
        except Exception as e:
            logger.warning("Probe failed for financial_metrics %s: %s", ticker, e)

    elif cached_data:
        # Have data but no report_period tracked — return cached
        return [FinancialMetrics(**m) for m in cached_data]

    # Full fetch
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    )
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        # If full fetch fails but we have stale cache, return it
        if cached_data:
            return [FinancialMetrics(**m) for m in cached_data]
        return []

    try:
        metrics_response = FinancialMetricsResponse(**response.json())
        financial_metrics = metrics_response.financial_metrics
    except Exception as e:
        logger.warning("Failed to parse financial metrics response for %s: %s", ticker, e)
        if cached_data:
            return [FinancialMetrics(**m) for m in cached_data]
        return []

    if not financial_metrics:
        return []

    # Determine latest report_period for probe tracking
    latest_rp = max(m.report_period for m in financial_metrics) if financial_metrics else None

    _cache.set_financial_metrics(
        cache_key,
        [m.model_dump() for m in financial_metrics],
        latest_report_period=latest_rp,
    )
    return financial_metrics


# ---------------------------------------------------------------------------
# Line items — PROBE-BASED (same strategy as financial_metrics)
# ---------------------------------------------------------------------------
def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items with probe-based smart caching."""
    # Build a stable cache key from sorted line_items
    items_key = "_".join(sorted(line_items))
    cache_key = f"{ticker}_{items_key}_{period}_{end_date}_{limit}"
    headers = _get_headers(api_key)

    cached_data = _cache.get_line_items(cache_key)
    cached_report_period = _cache.get_latest_report_period("line_items", cache_key)

    if cached_data and cached_report_period:
        # PROBE: use financial_metrics limit=1 to check for new report
        # (cheaper than line_items POST, and report_period is shared)
        probe_url = (
            f"https://api.financialdatasets.ai/financial-metrics/"
            f"?ticker={ticker}&report_period_lte={end_date}&limit=1&period={period}"
        )
        try:
            probe_resp = _make_api_request(probe_url, headers)
            if probe_resp.status_code == 200:
                probe_data = probe_resp.json()
                probe_metrics = probe_data.get("financial_metrics", [])
                if probe_metrics:
                    latest_rp = probe_metrics[0].get("report_period", "")
                    if latest_rp == cached_report_period:
                        logger.info(
                            "Smart cache HIT for line_items %s "
                            "(report_period unchanged: %s)",
                            ticker, latest_rp,
                        )
                        return [LineItem(**item) for item in cached_data]
                    else:
                        logger.info(
                            "Smart cache MISS for line_items %s "
                            "(new report_period: %s → %s)",
                            ticker, cached_report_period, latest_rp,
                        )
        except Exception as e:
            logger.warning("Probe failed for line_items %s: %s", ticker, e)

    elif cached_data:
        return [LineItem(**item) for item in cached_data]

    # Full fetch
    url = "https://api.financialdatasets.ai/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        if cached_data:
            return [LineItem(**item) for item in cached_data]
        return []

    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = response_model.search_results
    except Exception as e:
        logger.warning("Failed to parse line items response for %s: %s", ticker, e)
        if cached_data:
            return [LineItem(**item) for item in cached_data]
        return []

    if not search_results:
        return []

    results = search_results[:limit]

    # Determine latest report_period
    report_periods = [r.report_period for r in results if hasattr(r, "report_period") and r.report_period]
    latest_rp = max(report_periods) if report_periods else None

    _cache.set_line_items(
        cache_key,
        [r.model_dump() for r in results],
        latest_report_period=latest_rp,
    )
    return results


# ---------------------------------------------------------------------------
# Insider trades — 7-day time-based refresh
# ---------------------------------------------------------------------------
def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades with 7-day smart caching."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    # Check if cached data is still fresh (within 7 days)
    if _cache.is_fresh("insider_trades", cache_key):
        cached_data = _cache.get_insider_trades(cache_key)
        if cached_data:
            logger.info("Smart cache HIT for insider_trades %s (within 7-day window)", ticker)
            return [InsiderTrade(**trade) for trade in cached_data]

    headers = _get_headers(api_key)
    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        if not start_date or len(insider_trades) < limit:
            break

        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]
        if current_end_date <= start_date:
            break

    if not all_trades:
        # Return stale cache if API fails
        cached_data = _cache.get_insider_trades(cache_key)
        if cached_data:
            logger.info("Returning stale cache for insider_trades %s", ticker)
            return [InsiderTrade(**trade) for trade in cached_data]
        return []

    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


# ---------------------------------------------------------------------------
# Company news — 3-day time-based refresh
# ---------------------------------------------------------------------------
def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news with 3-day smart caching."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    # Check if cached data is still fresh (within 3 days)
    if _cache.is_fresh("company_news", cache_key):
        cached_data = _cache.get_company_news(cache_key)
        if cached_data:
            logger.info("Smart cache HIT for company_news %s (within 3-day window)", ticker)
            return [CompanyNews(**news) for news in cached_data]

    headers = _get_headers(api_key)
    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.warning("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        if not start_date or len(company_news) < limit:
            break

        current_end_date = min(news.date for news in company_news).split("T")[0]
        if current_end_date <= start_date:
            break

    if not all_news:
        # Return stale cache if API fails
        cached_data = _cache.get_company_news(cache_key)
        if cached_data:
            logger.info("Returning stale cache for company_news %s", ticker)
            return [CompanyNews(**news) for news in cached_data]
        return []

    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


# ---------------------------------------------------------------------------
# Market cap — try to derive from free price data first
# ---------------------------------------------------------------------------
def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap. Tries company facts first, falls back to financial_metrics."""
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        headers = _get_headers(api_key)
        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    # Fallback: extract from financial_metrics (which is now smartly cached)
    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap
    return market_cap if market_cap else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
