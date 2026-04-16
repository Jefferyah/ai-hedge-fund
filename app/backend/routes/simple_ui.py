"""
Ultra-simple UI + API for one-shot hedge-fund analysis.

Exposes:
  GET  /simple                 → dashboard HTML page
  GET  /simple/dashboard       → latest analysis per watchlist ticker (JSON)
  GET  /simple/history/{tkr}   → history of analyses for one ticker (JSON)
  POST /simple/analyze         → run a new analysis; persists to DB

The pipeline is hardcoded to: technical_analyst → risk_management → portfolio_manager
(the smallest useful flow that doesn't need any user-built graph).
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.backend.database import get_db
from app.backend.database.models import SimpleAnalysis
from app.backend.models.schemas import GraphEdge, GraphNode
from app.backend.services.api_key_service import ApiKeyService
from app.backend.services.graph import (
    create_graph,
    parse_hedge_fund_response,
    run_graph_async,
)
from app.backend.services.portfolio import create_portfolio

router = APIRouter(prefix="/simple")

# The five tickers the FinancialDatasets free tier supports. The dashboard
# page is hardcoded to show these; the analyze endpoint still accepts any
# ticker for one-offs.
WATCHLIST = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]


@router.get("/debug/fd")
async def debug_financial_datasets(
    ticker: str = "GOOGL",
    days: int = 180,
    db: Session = Depends(get_db),
):
    """Diagnostic: make raw HTTP call + call get_prices() the same way the
    agent does, to spot the divergence."""
    import requests
    from src.tools.api import get_prices

    keys = ApiKeyService(db).get_api_keys_dict()
    fd_key = keys.get("FINANCIAL_DATASETS_API_KEY")

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.utcnow().strftime("%Y-%m-%d")

    # --- Raw HTTP ---
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}&interval=day&interval_multiplier=1"
        f"&start_date={start}&end_date={end}"
    )
    headers = {"X-API-KEY": fd_key} if fd_key else {}
    raw = {}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        raw = {
            "status": r.status_code,
            "body_preview": r.text[:400],
            "num_rows": len((r.json() or {}).get("prices", [])) if r.status_code == 200 else 0,
        }
    except Exception as e:
        raw = {"error": f"{type(e).__name__}: {e}"}

    # --- via get_prices (the path the agent uses) ---
    agent_path = {}
    try:
        prices = get_prices(
            ticker=ticker, start_date=start, end_date=end, api_key=fd_key
        )
        agent_path = {
            "count": len(prices),
            "first": (prices[0].model_dump() if prices else None),
        }
    except Exception as e:
        agent_path = {"error": f"{type(e).__name__}: {e}"}

    return {
        "key_present": bool(fd_key),
        "key_prefix": (fd_key[:8] + "..." + fd_key[-4:]) if fd_key else None,
        "window": {"start": start, "end": end, "days": days},
        "raw_requests": raw,
        "via_get_prices": agent_path,
    }


class SimpleAnalyzeRequest(BaseModel):
    ticker: str
    model_name: Optional[str] = "llama-3.1-8b-instant"
    model_provider: Optional[str] = "Groq"


@router.post("/analyze")
async def simple_analyze(req: SimpleAnalyzeRequest, db: Session = Depends(get_db)):
    """Run a minimal one-ticker analysis and return the final decision as JSON."""
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    # Hardcoded minimal flow:
    #   technical_analyst_aaaaaa → portfolio_manager_bbbbbb
    # The graph builder auto-routes this through a risk manager.
    graph_nodes = [
        GraphNode(id="technical_analyst_aaaaaa", type="agent", data={}),
        GraphNode(id="portfolio_manager_bbbbbb", type="agent", data={}),
    ]
    graph_edges = [
        GraphEdge(
            id="e1",
            source="technical_analyst_aaaaaa",
            target="portfolio_manager_bbbbbb",
        ),
    ]

    # Date window: past 180 days, ending yesterday.
    # We deliberately use (today - 1) so that (a) today's bar — which isn't
    # closed yet — isn't requested, and (b) we don't race FinancialDatasets'
    # "today" (they use US market time; UTC may already be on the next day).
    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=181)).strftime("%Y-%m-%d")

    # Load API keys from DB
    api_keys = ApiKeyService(db).get_api_keys_dict()

    portfolio = create_portfolio(
        initial_cash=100000.0,
        margin_requirement=0.0,
        tickers=[ticker],
        portfolio_positions=None,
    )

    graph = create_graph(graph_nodes=graph_nodes, graph_edges=graph_edges).compile()

    # Build a minimal "request-like" object that run_graph_async expects
    # for agent-specific model access. We reuse HedgeFundRequest so the
    # agent functions find the model settings where they look for them.
    from app.backend.models.schemas import HedgeFundRequest

    hf_req = HedgeFundRequest(
        tickers=[ticker],
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        model_name=req.model_name,
        model_provider=req.model_provider,
        api_keys=api_keys,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100000.0,
    )

    try:
        result = await run_graph_async(
            graph=graph,
            portfolio=portfolio,
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            model_name=req.model_name,
            model_provider=req.model_provider,
            request=hf_req,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analysis failed: {e}")

    if not result or not result.get("messages"):
        raise HTTPException(status_code=500, detail="no result produced")

    decisions = parse_hedge_fund_response(result["messages"][-1].content)
    analyst_signals = result.get("data", {}).get("analyst_signals", {})
    current_prices = result.get("data", {}).get("current_prices", {})

    response = {
        "ticker": ticker,
        "window": {"start": start_date, "end": end_date},
        "model": f"{req.model_provider} / {req.model_name}",
        "decisions": decisions,
        "analyst_signals": analyst_signals,
        "current_prices": current_prices,
    }

    # Persist this analysis so the dashboard can show it without re-running.
    try:
        dec = (decisions or {}).get(ticker) or {}
        # current_price lives on the risk-management signal
        rm = (
            (analyst_signals or {})
            .get("risk_management_agent_bbbbbb", {})
            .get(ticker, {})
        )
        price = rm.get("current_price") if isinstance(rm, dict) else None
        row = SimpleAnalysis(
            ticker=ticker,
            model=f"{req.model_provider}/{req.model_name}",
            action=dec.get("action"),
            quantity=int(dec.get("quantity")) if dec.get("quantity") is not None else None,
            confidence=int(dec.get("confidence")) if dec.get("confidence") is not None else None,
            current_price=str(price) if price is not None else None,
            reasoning=dec.get("reasoning"),
            payload=response,
        )
        db.add(row)
        db.commit()
        response["saved_id"] = row.id
    except Exception:
        # Never let a DB write failure break a successful analysis.
        db.rollback()

    return response


def _row_to_dict(row: SimpleAnalysis) -> dict:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "model": row.model,
        "action": row.action,
        "quantity": row.quantity,
        "confidence": row.confidence,
        "current_price": float(row.current_price) if row.current_price else None,
        "reasoning": row.reasoning,
        "payload": row.payload,
    }


@router.get("/dashboard")
async def dashboard(db: Session = Depends(get_db)):
    """Return the latest analysis per watchlist ticker.

    Response shape: { tickers: { AAPL: {...} | null, ... } }
    Tickers with no analysis yet map to null so the frontend can render
    an empty placeholder card.
    """
    result = {}
    for t in WATCHLIST:
        row = (
            db.query(SimpleAnalysis)
            .filter(SimpleAnalysis.ticker == t)
            .order_by(SimpleAnalysis.created_at.desc())
            .first()
        )
        result[t] = _row_to_dict(row) if row else None
    return {"watchlist": WATCHLIST, "tickers": result}


@router.get("/history/{ticker}")
async def history(ticker: str, limit: int = 20, db: Session = Depends(get_db)):
    """Return the last N analyses for a single ticker, newest first."""
    ticker = ticker.strip().upper()
    limit = max(1, min(limit, 100))
    rows = (
        db.query(SimpleAnalysis)
        .filter(SimpleAnalysis.ticker == ticker)
        .order_by(SimpleAnalysis.created_at.desc())
        .limit(limit)
        .all()
    )
    # Strip the heavy `payload` field from history entries — the list view
    # only needs the headline fields. The detail view re-fetches via /dashboard.
    items = []
    for r in rows:
        d = _row_to_dict(r)
        d.pop("payload", None)
        items.append(d)
    return {"ticker": ticker, "count": len(items), "items": items}


_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Hedge Fund — Dashboard</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #181b22;
    --panel-2: #1f232d;
    --border: #2a2f3a;
    --text: #e7e9ee;
    --muted: #8a92a3;
    --accent: #4f8cff;
    --buy: #22c55e;
    --sell: #ef4444;
    --hold: #eab308;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", sans-serif; }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 32px 24px 80px; }

  /* Header */
  header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
  h1 { font-size: 22px; margin: 0; font-weight: 600; flex: 1; }
  .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .last-sync { color: var(--muted); font-size: 12px; }
  select, button {
    font: inherit; color: var(--text); background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 9px 13px; font-size: 13px;
  }
  select:focus { outline: none; border-color: var(--accent); }
  button { cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  button.ghost { background: transparent; }
  button.ghost:hover { border-color: var(--accent); }
  .sub { color: var(--muted); font-size: 13px; margin: 4px 0 24px; }

  /* Grid of cards */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px; }

  /* Card */
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 18px 16px; display: flex; flex-direction: column; gap: 10px;
    transition: border-color 0.15s; }
  .card:hover { border-color: #3a4050; }
  .card.loading { opacity: 0.55; }
  .card .head { display: flex; align-items: center; gap: 10px; }
  .card .ticker { font-size: 20px; font-weight: 700; letter-spacing: 0.5px; flex: 1; }
  .card .ref-btn { background: transparent; border: 1px solid var(--border); color: var(--muted);
    width: 30px; height: 30px; padding: 0; border-radius: 6px; font-size: 14px; }
  .card .ref-btn:hover { color: var(--accent); border-color: var(--accent); }
  .card .decision-line { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .card .qty { font-size: 13px; color: var(--muted); }
  .card .price { font-size: 20px; font-weight: 600; margin-left: auto; }
  .card .conf { color: var(--muted); font-size: 12px; }
  .card .stamp { color: var(--muted); font-size: 11px; }
  .card .empty { color: var(--muted); font-size: 13px; padding: 12px 0; text-align: center; }
  .card .empty button { margin-top: 8px; }

  /* Mini strat chips */
  .mini-strats { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
  .mini-chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 8px;
    border-radius: 999px; background: var(--panel-2); border: 1px solid var(--border);
    font-size: 11px; color: var(--muted); }
  .mini-chip .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
  .mini-chip.bullish .dot { background: var(--buy); }
  .mini-chip.bearish .dot { background: var(--sell); }
  .mini-chip.neutral .dot { background: var(--muted); }

  /* Badges */
  .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700;
    font-size: 12px; letter-spacing: 0.5px; }
  .badge.buy, .badge.cover, .badge.bullish { background: rgba(34,197,94,0.15); color: var(--buy); }
  .badge.sell, .badge.short, .badge.bearish { background: rgba(239,68,68,0.15); color: var(--sell); }
  .badge.hold { background: rgba(234,179,8,0.15); color: var(--hold); }
  .badge.neutral { background: rgba(138,146,163,0.15); color: var(--muted); }
  .badge.sm { font-size: 10px; padding: 2px 8px; }

  /* Expand button */
  .expand { background: transparent; border: none; color: var(--muted); font-size: 12px;
    padding: 4px 0; margin-top: 4px; cursor: pointer; text-align: left; }
  .expand:hover { color: var(--accent); }

  /* Expanded detail */
  .detail { border-top: 1px dashed var(--border); padding-top: 12px; margin-top: 6px;
    display: none; font-size: 13px; }
  .card.open .detail { display: block; }
  .detail .section-title { font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.8px; margin: 14px 0 8px; }
  .detail .section-title:first-child { margin-top: 0; }
  .strat-grid { display: grid; grid-template-columns: max-content auto;
    gap: 5px 12px; font-size: 12px; }
  .strat-grid .sk { color: var(--muted); }
  .stat-line { color: var(--muted); font-size: 12px; line-height: 1.7; }
  .reasoning-text { color: var(--text); font-size: 12px; line-height: 1.6; }

  /* History list */
  .history-list { display: flex; flex-direction: column; gap: 4px; max-height: 180px;
    overflow-y: auto; }
  .history-row { display: flex; gap: 10px; font-size: 11px; padding: 4px 0;
    border-bottom: 1px dotted var(--border); align-items: center; }
  .history-row:last-child { border-bottom: none; }
  .history-row .h-time { color: var(--muted); min-width: 90px; }
  .history-row .h-action { min-width: 52px; }
  .history-row .h-conf { color: var(--muted); margin-left: auto; }

  /* Tooltips — hover over any [data-tip] element */
  [data-tip] { border-bottom: 1px dotted rgba(138,146,163,0.4); cursor: help; }
  [data-tip]:hover { border-bottom-color: var(--accent); }
  [data-tip]::after {
    content: attr(data-tip);
    position: absolute; left: 50%; transform: translateX(-50%) translateY(-8px);
    background: #1f232d; border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 12px; font-size: 12px; line-height: 1.55; color: var(--text);
    width: 260px; white-space: normal; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    opacity: 0; pointer-events: none; transition: opacity 0.12s; z-index: 100;
    margin-top: -4px;
  }
  [data-tip] { position: relative; }
  [data-tip]:hover::after { opacity: 1; transform: translateX(-50%) translateY(-100%); }

  /* Spinner */
  .spin { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid var(--border); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .err { color: #fca5a5; font-size: 12px; padding: 8px 10px;
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25);
    border-radius: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>AI Hedge Fund — Dashboard</h1>
    <div class="controls">
      <select id="model">
        <option value="llama-3.1-8b-instant">Llama 3.1 8B (快)</option>
        <option value="llama-3.3-70b-versatile">Llama 3.3 70B (深)</option>
        <option value="moonshotai/kimi-k2-instruct">Kimi K2</option>
        <option value="deepseek-r1-distill-llama-70b">DeepSeek R1 Distill</option>
      </select>
      <button id="refreshAll" class="primary">全部刷新</button>
    </div>
  </header>
  <div class="sub">
    AAPL / GOOGL / MSFT / NVDA / TSLA 的 180 天
    <span data-tip="只看歷史股價和成交量的走勢，不看財報、產業新聞或公司基本面。">技術分析</span>。
    打開頁面會顯示上次跑過的結果；按「全部刷新」或單張卡的 ↻ 才會重新分析。
    滑鼠移到有虛線的詞彙上會有白話說明。
  </div>

  <div class="grid" id="grid"></div>
</div>

<script>
const TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"];

const STRAT_LABEL = {
  trend_following: "趨勢追蹤",
  mean_reversion: "均值回歸",
  momentum: "動能",
  volatility: "波動率",
  statistical_arbitrage: "統計套利",
};

const STRAT_TIP = {
  trend_following: "順勢操作：股價一路漲就看多、一路跌就看空。看的是有沒有明顯的方向。",
  mean_reversion: "反向操作：股價偏離平均太多就預期會拉回。看的是有沒有被過度炒作或錯殺。",
  momentum: "看最近的漲跌速度。最近 1 / 3 / 6 個月漲多少、跌多少。",
  volatility: "看股價震盪幅度。震盪變大通常代表市場有事，傾向看空。",
  statistical_arbitrage: "分析股價的統計特性（分布偏度、長期記憶性），從異常偏差中找交易機會。",
};

const ACTION_LABEL = { buy: "買入", sell: "賣出", hold: "觀望", short: "放空", cover: "回補" };
const ACTION_TIP = {
  buy: "預期股價會上漲，建議買進這個數量的股票。",
  sell: "預期股價會下跌，建議賣掉手上的持股。",
  hold: "訊號不明確，建議不動作、繼續觀察。",
  short: "預期股價會下跌，建議借來賣（賣高再買回）。風險比一般買入高。",
  cover: "買回先前放空借出的股票，結清空頭部位。",
};

const SIGNAL_LABEL = { bullish: "看多", bearish: "看空", neutral: "中性" };
const SIGNAL_TIP = {
  bullish: "預期上漲。",
  bearish: "預期下跌。",
  neutral: "沒有明顯方向。",
};

const CONF_TIP = "AI 對這個決策的把握程度。100% 代表非常確定，20% 代表只是稍微偏向這個方向。";
const PRICE_TIP = "最近一根日 K 的收盤價。";
const LIMIT_TIP = "風險管理允許這檔股票最多還能再投入多少錢，用來避免單檔押太重。";
const VOL_TIP = "把日波動放大成一年的尺度。20% 左右是大盤正常水準，30-40% 偏高，50%+ 很高。";
const PCTL_TIP = "把這檔股票目前的波動率，拿來跟它過去的波動率比。50 分位 = 中間、80 分位 = 比過去 80% 的時間都波動。";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function badge(action, sm) {
  const a = (action || "").toLowerCase();
  const label = ACTION_LABEL[a] || SIGNAL_LABEL[a] || action || "—";
  const tip = ACTION_TIP[a] || SIGNAL_TIP[a] || "";
  const sp = sm ? " sm" : "";
  return tip
    ? `<span class="badge ${a}${sp}" data-tip="${esc(tip)}">${label}</span>`
    : `<span class="badge ${a}${sp}">${label}</span>`;
}

function timeAgo(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const s = Math.max(1, Math.floor((Date.now() - d.getTime()) / 1000));
  if (s < 60) return s + " 秒前";
  if (s < 3600) return Math.floor(s / 60) + " 分鐘前";
  if (s < 86400) return Math.floor(s / 3600) + " 小時前";
  return Math.floor(s / 86400) + " 天前";
}

function formatWhen(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("zh-TW", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function cardHtml(ticker, row) {
  if (!row) {
    return `
      <div class="card" data-ticker="${ticker}">
        <div class="head">
          <span class="ticker">${ticker}</span>
          <button class="ref-btn" title="分析這一檔" onclick="analyzeOne('${ticker}')">↻</button>
        </div>
        <div class="empty">尚無資料<br><button class="ghost" onclick="analyzeOne('${ticker}')">立即分析</button></div>
      </div>`;
  }
  const payload = row.payload || {};
  const ta = ((payload.analyst_signals || {})["technical_analyst_aaaaaa"] || {})[ticker] || {};
  const rm = ((payload.analyst_signals || {})["risk_management_agent_bbbbbb"] || {})[ticker] || {};
  const strats = ta.reasoning && typeof ta.reasoning === "object" ? ta.reasoning : {};

  let miniChips = "";
  for (const k of Object.keys(STRAT_LABEL)) {
    const s = strats[k];
    const sig = (s && s.signal) || "neutral";
    miniChips += `<span class="mini-chip ${sig}" data-tip="${esc(STRAT_LABEL[k] + '：' + (STRAT_TIP[k] || ''))}">`
      + `<span class="dot"></span>${STRAT_LABEL[k]}</span>`;
  }

  // Expanded detail
  let stratGrid = "";
  for (const [k, s] of Object.entries(strats)) {
    stratGrid += `<div class="sk" data-tip="${esc(STRAT_TIP[k] || '')}">${STRAT_LABEL[k] || k}</div>`;
    stratGrid += `<div>${badge(s.signal, true)} <span style="color:var(--muted)">${s.confidence ?? 0}%</span></div>`;
  }
  const vol = rm.volatility_metrics || {};
  const riskLines = [];
  if (rm.current_price != null)
    riskLines.push(`<span data-tip="${PRICE_TIP}">現價</span> $${Number(rm.current_price).toFixed(2)}`);
  if (rm.remaining_position_limit != null)
    riskLines.push(`<span data-tip="${LIMIT_TIP}">剩餘倉位上限</span> $${Math.round(Number(rm.remaining_position_limit)).toLocaleString()}`);
  if (vol.annualized_volatility != null)
    riskLines.push(`<span data-tip="${VOL_TIP}">年化波動</span> ${(vol.annualized_volatility * 100).toFixed(1)}%`);
  if (vol.volatility_percentile != null)
    riskLines.push(`<span data-tip="${PCTL_TIP}">波動位階</span> ${vol.volatility_percentile.toFixed(0)} 分位`);

  return `
    <div class="card" data-ticker="${ticker}">
      <div class="head">
        <span class="ticker">${ticker}</span>
        <button class="ref-btn" title="刷新這一檔" onclick="analyzeOne('${ticker}')">↻</button>
      </div>
      <div class="decision-line">
        ${badge(row.action)}
        ${row.quantity != null ? `<span class="qty">${row.quantity} 股</span>` : ""}
        ${row.current_price != null ? `<span class="price">$${Number(row.current_price).toFixed(2)}</span>` : ""}
      </div>
      <div class="conf"><span data-tip="${CONF_TIP}">信心度</span> ${row.confidence ?? 0}%</div>
      <div class="mini-strats">${miniChips}</div>
      <div class="stamp">${timeAgo(row.created_at)} · ${esc(row.model || "")}</div>
      <button class="expand" onclick="toggleCard('${ticker}')">展開細節 / 歷史 ▾</button>
      <div class="detail">
        <div class="section-title">技術分析五策略</div>
        <div class="strat-grid">${stratGrid}</div>
        <div class="section-title">風險管理</div>
        <div class="stat-line">${riskLines.join(" · ")}</div>
        ${row.reasoning ? `<div class="section-title">Portfolio Manager 理由</div><div class="reasoning-text">${esc(row.reasoning)}</div>` : ""}
        <div class="section-title">歷史記錄</div>
        <div class="history-list" id="hist-${ticker}"><div class="stat-line">載入中…</div></div>
      </div>
    </div>`;
}

async function loadDashboard() {
  const resp = await fetch("/simple/dashboard");
  const data = await resp.json();
  const grid = $("grid");
  grid.innerHTML = "";
  for (const t of TICKERS) {
    grid.insertAdjacentHTML("beforeend", cardHtml(t, data.tickers[t]));
  }
}

async function loadHistory(ticker) {
  const el = document.getElementById(`hist-${ticker}`);
  if (!el) return;
  try {
    const resp = await fetch(`/simple/history/${ticker}?limit=20`);
    const data = await resp.json();
    if (!data.items.length) {
      el.innerHTML = `<div class="stat-line">還沒有歷史紀錄</div>`;
      return;
    }
    el.innerHTML = data.items.map(it => `
      <div class="history-row">
        <span class="h-time">${formatWhen(it.created_at)}</span>
        <span class="h-action">${badge(it.action, true)}</span>
        <span class="stat-line">${it.quantity ?? 0} 股 · $${it.current_price != null ? Number(it.current_price).toFixed(2) : "—"}</span>
        <span class="h-conf">${it.confidence ?? 0}%</span>
      </div>
    `).join("");
  } catch (e) {
    el.innerHTML = `<div class="err">歷史載入失敗</div>`;
  }
}

function toggleCard(ticker) {
  const card = document.querySelector(`.card[data-ticker="${ticker}"]`);
  if (!card) return;
  const wasOpen = card.classList.contains("open");
  card.classList.toggle("open");
  const btn = card.querySelector(".expand");
  if (btn) btn.textContent = wasOpen ? "展開細節 / 歷史 ▾" : "收起 ▴";
  if (!wasOpen) loadHistory(ticker);
}

async function analyzeOne(ticker) {
  const card = document.querySelector(`.card[data-ticker="${ticker}"]`);
  if (!card) return;
  card.classList.add("loading");
  const refBtn = card.querySelector(".ref-btn");
  if (refBtn) refBtn.innerHTML = `<span class="spin"></span>`;
  try {
    const resp = await fetch("/simple/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, model_name: $("model").value, model_provider: "Groq" })
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || String(resp.status));
    }
    // Just reload the whole dashboard — simpler than patching one card
    await loadDashboard();
  } catch (e) {
    alert(`${ticker} 分析失敗：${e.message || e}`);
    card.classList.remove("loading");
    if (refBtn) refBtn.textContent = "↻";
  }
}

async function refreshAll() {
  const btn = $("refreshAll");
  btn.disabled = true;
  btn.textContent = "分析中…";
  // Fire all 5 in parallel. Each takes ~15-30s; total ≈ slowest one.
  try {
    await Promise.allSettled(TICKERS.map(t =>
      fetch("/simple/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: t, model_name: $("model").value, model_provider: "Groq" })
      })
    ));
    await loadDashboard();
  } finally {
    btn.disabled = false;
    btn.textContent = "全部刷新";
  }
}

$("refreshAll").addEventListener("click", refreshAll);
loadDashboard();
</script>
</body>
</html>
"""


@router.get("", response_class=HTMLResponse)
async def simple_page():
    return HTMLResponse(_HTML)
