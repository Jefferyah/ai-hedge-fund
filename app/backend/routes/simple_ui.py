"""
Ultra-simple UI + API for one-shot hedge-fund analysis.

Exposes:
  GET  /simple                 → dashboard HTML page
  GET  /simple/dashboard       → latest analysis per watchlist ticker (JSON)
  GET  /simple/history/{tkr}   → history of analyses for one ticker (JSON)
  GET  /simple/history-all     → history of analyses for all tickers (JSON)
  POST /simple/analyze         → run a new analysis; persists to DB

The pipeline is hardcoded to: technical_analyst → risk_management → portfolio_manager
(the smallest useful flow that doesn't need any user-built graph).

Access control: if SIGNALDECK_PASSCODE env var is set, all mutating endpoints
(analyze) require an X-Passcode header or ?passcode= query param matching it.
The dashboard/history read endpoints are always open so cached data loads fast.
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query
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

# The five tickers the FinancialDatasets free tier supports.
WATCHLIST = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]

_PASSCODE = os.getenv("SIGNALDECK_PASSCODE", "")


def _check_passcode(
    x_passcode: Optional[str] = Header(None),
    passcode: Optional[str] = Query(None),
):
    """Dependency that enforces passcode when SIGNALDECK_PASSCODE is set."""
    if not _PASSCODE:
        return  # no passcode configured → open access
    supplied = x_passcode or passcode or ""
    if supplied != _PASSCODE:
        raise HTTPException(status_code=403, detail="passcode_required")


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


@router.post("/analyze", dependencies=[Depends(_check_passcode)])
async def simple_analyze(req: SimpleAnalyzeRequest, db: Session = Depends(get_db)):
    """Run a minimal one-ticker analysis and return the final decision as JSON."""
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    # Analyst nodes — all run in parallel, then feed risk_mgr → portfolio_mgr
    analyst_ids = [
        "technical_analyst_aaaaaa",        # 技術分析（純價格）
        "nassim_taleb_aaaaab",             # 塔雷伯：黑天鵝風控
        "michael_burry_aaaaac",            # 乃大空頭：逆向深度價值
        "warren_buffett_aaaaad",           # 巴菲特：價值投資
        "cathie_wood_aaaaae",              # 木頭姐：破壞式創新
        "stanley_druckenmiller_aaafff",    # 朱肯米勒：宏觀趨勢
        "ben_graham_aaaaag",               # 葛拉漢：安全邊際
        "bill_ackman_aaaaah",              # 阿克曼：行動派價值
        "charlie_munger_aaaajj",           # 蒙格：護城河
        "peter_lynch_aaaakk",              # 彼得林區：成長股
        "phil_fisher_aaaall",              # 費雪：質化分析
        "mohnish_pabrai_aaaamm",           # 帕布萊：集中投資
        "rakesh_jhunjhunwala_aaaann",      # 印度巴菲特
        "aswath_damodaran_aaaaoo",         # 達摩德蘭：學術估值
        "fundamentals_analyst_aaaapp",     # 基本面分析師
        "valuation_analyst_aaaaqq",        # 估值分析師
        "growth_analyst_aaaarr",           # 成長分析師
        "sentiment_analyst_aaaass",        # 情緒分析師
        "news_sentiment_analyst_aaaatt",   # 新聞情緒分析師
    ]
    pm_id = "portfolio_manager_bbbbbb"

    graph_nodes = [GraphNode(id=a, type="agent", data={}) for a in analyst_ids]
    graph_nodes.append(GraphNode(id=pm_id, type="agent", data={}))

    graph_edges = [
        GraphEdge(id=f"e{i}", source=a, target=pm_id)
        for i, a in enumerate(analyst_ids)
    ]

    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=181)).strftime("%Y-%m-%d")

    api_keys = ApiKeyService(db).get_api_keys_dict()

    portfolio = create_portfolio(
        initial_cash=100000.0,
        margin_requirement=0.0,
        tickers=[ticker],
        portfolio_positions=None,
    )

    graph = create_graph(graph_nodes=graph_nodes, graph_edges=graph_edges).compile()

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
            language="繁體中文",
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

    try:
        dec = (decisions or {}).get(ticker) or {}
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
    """Return the latest analysis per watchlist ticker."""
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
    items = []
    for r in rows:
        d = _row_to_dict(r)
        d.pop("payload", None)
        items.append(d)
    return {"ticker": ticker, "count": len(items), "items": items}


@router.get("/history-all")
async def history_all(limit: int = 30, db: Session = Depends(get_db)):
    """Return last N analyses for every watchlist ticker in one call."""
    limit = max(1, min(limit, 100))
    result = {}
    for t in WATCHLIST:
        rows = (
            db.query(SimpleAnalysis)
            .filter(SimpleAnalysis.ticker == t)
            .order_by(SimpleAnalysis.created_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for r in rows:
            d = _row_to_dict(r)
            d.pop("payload", None)
            items.append(d)
        result[t] = items
    return {"tickers": result}


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SignalDeck</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root {
    --bg: #191919; --panel: #202020; --panel-2: #1f1f1f;
    --border: #2a2a2a; --border-light: #2e2e2e;
    --text: #e8e8e8; --muted: #888888; --dim: #555555;
    --accent: #8b8bf5; --accent-dim: rgba(139,139,245,.10);
    --buy: #22c55e; --sell: #ef4444; --hold: #eab308;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", "Noto Sans TC", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased; font-size: 14px; line-height: 1.5; }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 32px 24px 56px; }

  /* ---- Header ---- */
  header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 4px;
    padding-bottom: 16px; border-bottom: 1px solid var(--border); }
  .logo { font-size: 18px; font-weight: 700; letter-spacing: -.3px; flex: 1;
    color: var(--text); }
  .logo .logo-dot { color: var(--accent); }
  select, button { font: inherit; font-size: 13px; color: var(--muted); background: transparent;
    border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; cursor: pointer;
    transition: border-color .15s, color .15s; }
  select:hover, button:hover { color: var(--text); border-color: var(--border-light); }
  select:focus, button:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-dim); }
  .primary { background: var(--accent); border: none;
    color: #fff; font-weight: 600; padding: 8px 18px; border-radius: 8px;
    box-shadow: none; }
  .primary:hover { background: #7a7ae8; box-shadow: none; }
  .primary:disabled { opacity: .4; cursor: not-allowed; }
  .sub { color: var(--muted); font-size: 12px; margin-top: 14px; margin-bottom: 24px; line-height: 1.7; }

  /* ---- Tooltip ---- */
  [data-tip] { border-bottom: 1px dotted var(--dim); cursor: help; position: relative; }
  [data-tip]:hover { border-bottom-color: var(--accent); }
  [data-tip]::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 10px);
    left: 50%; transform: translateX(-50%); background: #282828;
    border: 1px solid var(--border-light); border-radius: 8px; padding: 10px 14px;
    font-size: 12px; line-height: 1.55; color: var(--text); width: 260px; white-space: normal;
    box-shadow: 0 4px 16px rgba(0,0,0,.4); opacity: 0; pointer-events: none;
    transition: opacity .15s; z-index: 100; }
  [data-tip]:hover::after { opacity: 1; }

  /* ---- Panels ---- */
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px 22px 18px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.3); }
  .panel-title { font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 14px; font-weight: 600; }

  /* ---- Signal bar chart ---- */
  .signal-wrap { position: relative; height: 250px; }

  /* ---- Data table ---- */
  .tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
  .tbl th { text-align: left; color: var(--dim); font-weight: 600; padding: 8px 12px;
    border-bottom: 1px solid var(--border); font-size: 10px; text-transform: uppercase;
    letter-spacing: .8px; }
  .tbl td { padding: 12px 12px; border-bottom: 1px solid var(--border);
    vertical-align: middle; transition: background .1s; }
  .tbl tr:last-child td { border-bottom: none; }
  .tbl tbody tr:hover td { background: rgba(255,255,255,.02); }
  .tbl .tk { font-weight: 600; font-size: 13px; letter-spacing: .3px; }
  .tbl .price { font-weight: 500; font-variant-numeric: tabular-nums; }
  .tbl .qty { font-variant-numeric: tabular-nums; }
  .tbl .conf-wrap { display: flex; align-items: center; gap: 6px; }
  .tbl .conf-bar { height: 5px; border-radius: 3px; min-width: 4px; transition: width .3s; }
  .tbl .conf-num { font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }
  .tbl .reasoning { color: var(--muted); font-size: 11px; max-width: 180px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tbl .stamp { color: var(--dim); font-size: 11px; }
  .ref-btn { background: transparent; border: 1px solid var(--border); color: var(--dim);
    width: 28px; height: 28px; padding: 0; border-radius: 6px; font-size: 13px;
    cursor: pointer; transition: all .15s; }
  .ref-btn:hover { color: var(--accent); border-color: var(--accent); }

  /* ---- Badges ---- */
  .badge { display: inline-block; padding: 3px 8px; border-radius: 4px;
    font-weight: 600; font-size: 11px; letter-spacing: .2px; white-space: nowrap; }
  .badge.buy, .badge.cover { background: rgba(34,197,94,.10); color: var(--buy); }
  .badge.sell, .badge.short { background: rgba(239,68,68,.10); color: var(--sell); }
  .badge.hold { background: rgba(234,179,8,.10); color: var(--hold); }
  .badge.bullish { background: rgba(34,197,94,.10); color: var(--buy); }
  .badge.bearish { background: rgba(239,68,68,.10); color: var(--sell); }
  .badge.neutral { background: rgba(136,136,136,.10); color: var(--muted); }

  /* ---- Clickable rows ---- */
  .tbl tbody tr { cursor: pointer; }
  .tbl tbody tr.active td { background: var(--accent-dim); }

  /* ---- Detail panel ---- */
  .detail-row td { padding: 0 !important; }
  .detail-panel { padding: 20px; background: var(--panel-2); border-top: 1px solid var(--border); }
  .detail-panel .dp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .detail-panel .dp-grid { grid-template-columns: 1fr; } }
  .detail-panel .dp-title { font-size: 10px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .8px; margin-bottom: 10px; font-weight: 600; }
  .detail-panel .strat-grid { display: grid; grid-template-columns: max-content auto;
    gap: 6px 16px; font-size: 12px; }
  .detail-panel .strat-grid .sk { color: var(--muted); }
  .detail-panel .stat-line { color: var(--muted); font-size: 12px; line-height: 2; }
  .detail-panel .reason-full { font-size: 12px; line-height: 1.7; margin-top: 6px;
    color: var(--text); padding: 10px 12px; background: rgba(0,0,0,.25); border-radius: 8px;
    border: 1px solid var(--border); }
  .analyst-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
  .analyst-card { background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px; }
  .analyst-hdr { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
  .analyst-name { font-weight: 500; font-size: 13px; color: var(--text); }
  .conf-sm { font-size: 11px; color: var(--muted); }
  .reason-text { font-size: 11px; color: var(--muted); line-height: 1.6; margin-top: 4px; }

  /* ---- History chart ---- */
  .history-wrap { position: relative; height: 280px; }

  /* ---- Passcode gate ---- */
  .gate { position: fixed; inset: 0; background: var(--bg); z-index: 999;
    display: flex; align-items: center; justify-content: center; }
  .gate-box { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 32px 28px; text-align: center; width: 340px;
    box-shadow: 0 4px 24px rgba(0,0,0,.35); }
  .gate-box .logo { font-size: 22px; margin-bottom: 8px; display: block; }
  .gate-box p { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .gate-box input { width: 100%; font: inherit; font-size: 16px; text-align: center;
    letter-spacing: 4px; padding: 10px; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px; }
  .gate-box input:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-dim); }
  .gate-box .err-msg { color: var(--sell); font-size: 12px; margin-top: 10px; min-height: 18px; }
  .gate-box button { margin-top: 14px; width: 100%; }

  /* ---- Misc ---- */
  .spin { display: inline-block; width: 13px; height: 13px; border-radius: 50%;
    border: 2px solid var(--border); border-top-color: var(--accent);
    animation: spin .8s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .empty-state { text-align: center; color: var(--muted); padding: 36px 0; font-size: 13px; }
  .empty-state button { margin-top: 12px; }
  .hidden { display: none !important; }
</style>
</head>
<body>

<!-- Passcode gate (only shown if SIGNALDECK_PASSCODE is set) -->
<div class="gate hidden" id="gate">
  <div class="gate-box">
    <span class="logo">Signal<span class="logo-dot">.</span>Deck</span>
    <p>請輸入通行碼</p>
    <input id="passInput" type="password" placeholder="Passcode" autocomplete="off" autofocus />
    <div class="err-msg" id="passErr"></div>
    <button class="primary" id="passBtn" onclick="submitPass()">進入</button>
  </div>
</div>

<div class="wrap" id="main">

  <header>
    <span class="logo">Signal<span class="logo-dot">.</span>Deck</span>
    <select id="model">
      <option value="llama-3.1-8b-instant">Llama 3.1 8B (快)</option>
      <option value="llama-3.3-70b-versatile">Llama 3.3 70B (深)</option>
      <option value="moonshotai/kimi-k2-instruct">Kimi K2</option>
      <option value="deepseek-r1-distill-llama-70b">DeepSeek R1 Distill</option>
    </select>
    <button class="primary" id="refreshAll">全部刷新</button>
  </header>
  <div class="sub">
    每天美東 17:05 自動更新 · 打開頁面顯示上次快照 · 按「全部刷新」立刻重跑 ·
    <span data-tip="AI 假設你有 $100,000 虛擬資金，根據 180 天技術分析（股價走勢，不看財報）決定要買還是賣、買幾股。不是投資建議。">滑鼠移到虛線文字看說明</span>
  </div>

  <!-- Signal Overview -->
  <div class="panel">
    <div class="panel-title">
      <span data-tip="橫向柱狀圖：往右 = 建議買入（綠色），往左 = 建議賣出或放空（紅色），長度 = AI 的信心程度。">訊號總覽</span>
    </div>
    <div class="signal-wrap"><canvas id="signalChart"></canvas></div>
  </div>

  <!-- Data Table -->
  <div class="panel">
    <div class="panel-title">明細 <span style="font-weight:400;text-transform:none;letter-spacing:0">— 點擊任一列展開詳細分析</span></div>
    <table class="tbl">
      <thead>
        <tr>
          <th>代號</th>
          <th data-tip="最近一根日 K 的收盤價。">現價</th>
          <th data-tip="AI 的交易建議。買入 = 預期上漲；賣出/放空 = 預期下跌；觀望 = 訊號不明確。">動作</th>
          <th data-tip="基於 $100,000 虛擬資金，風險管理計算出的建議部位大小（不是真錢）。">股數</th>
          <th data-tip="AI 對這個決策的把握程度。100% = 非常確定，20% = 只是稍微偏向。">信心</th>
          <th>理由</th>
          <th>更新</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="tableBody">
        <tr><td colspan="8" class="empty-state">載入中…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- History Trend Chart -->
  <div class="panel">
    <div class="panel-title">
      <span data-tip="Y 軸 = 方向性信心度：正值代表看多（買入），負值代表看空（賣出/放空）。五檔股票各一條線。">歷史走勢</span>
    </div>
    <div class="history-wrap"><canvas id="historyChart"></canvas></div>
    <div id="histEmpty" class="empty-state hidden">
      還沒有歷史資料。按「全部刷新」跑第一次分析後就會出現。
    </div>
  </div>

</div>

<script>
/* ===== Helpers ===== */
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* ===== Passcode ===== */
let storedPass = sessionStorage.getItem("sd_pass") || "";

// On load: try a test fetch. If 403, show the gate.
async function checkAccess() {
  try {
    const resp = await fetch("/simple/analyze", {
      method: "POST",
      headers: {"Content-Type":"application/json", ...(storedPass ? {"X-Passcode": storedPass} : {})},
      body: JSON.stringify({ticker:"TEST_AUTH_CHECK"}),
    });
    if (resp.status === 403) { showGate(); return false; }
    // Any other status (400 for bad ticker, etc.) means auth passed
    return true;
  } catch { return true; } // network error = no passcode enforced
}

function showGate() {
  $("gate").classList.remove("hidden");
  $("main").classList.add("hidden");
  setTimeout(() => $("passInput").focus(), 100);
}

async function submitPass() {
  const code = $("passInput").value.trim();
  if (!code) return;
  $("passBtn").disabled = true;
  try {
    const resp = await fetch("/simple/analyze", {
      method: "POST",
      headers: {"Content-Type":"application/json", "X-Passcode": code},
      body: JSON.stringify({ticker:"TEST_AUTH_CHECK"}),
    });
    if (resp.status === 403) {
      $("passErr").textContent = "通行碼錯誤";
      $("passBtn").disabled = false;
      return;
    }
    // Auth passed
    storedPass = code;
    sessionStorage.setItem("sd_pass", code);
    $("gate").classList.add("hidden");
    $("main").classList.remove("hidden");
    loadDashboard();
    renderHistoryChart();
  } catch (e) {
    $("passErr").textContent = "連線失敗";
    $("passBtn").disabled = false;
  }
}

$("passInput")?.addEventListener("keydown", e => { if (e.key === "Enter") submitPass(); });

/* ===== Constants ===== */
const TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"];
const TICKER_COLOR = {
  AAPL: "#5b9aff", GOOGL: "#34d399", MSFT: "#a78bfa", NVDA: "#22d3ee", TSLA: "#f87171",
};
const ACTION_LABEL = {buy:"買入",sell:"賣出",hold:"觀望",short:"放空",cover:"回補"};

function badge(action) {
  const a = (action||"").toLowerCase();
  return `<span class="badge ${a}">${ACTION_LABEL[a]||action||"—"}</span>`;
}
function timeAgo(iso) {
  if (!iso) return "—";
  const s = Math.max(1, Math.floor((Date.now() - new Date(iso).getTime())/1000));
  if (s<60) return s+"秒前"; if (s<3600) return Math.floor(s/60)+"分前";
  if (s<86400) return Math.floor(s/3600)+"時前"; return Math.floor(s/86400)+"天前";
}
function signedConf(action, conf) {
  const a = (action||"").toLowerCase();
  if (a==="buy"||a==="cover") return (conf||0);
  if (a==="sell"||a==="short") return -(conf||0);
  return 0;
}
function authHeaders() {
  const h = {"Content-Type":"application/json"};
  if (storedPass) h["X-Passcode"] = storedPass;
  return h;
}

/* ===== Signal Overview bar chart ===== */
let signalChartInst = null;
function renderSignalChart(tickers) {
  const ctx = $("signalChart");
  if (signalChartInst) signalChartInst.destroy();

  const entries = TICKERS.map(t => {
    const d = tickers[t];
    const raw = d ? signedConf(d.action, d.confidence) : 0;
    const action = d ? (d.action||"").toLowerCase() : "";
    const isHold = action === "hold" || (!action && d);
    return { ticker: t, sc: raw, data: d, isHold };
  }).sort((a,b) => a.sc - b.sc);

  // Hold/neutral: show a small bar so it's visible (yellow)
  const chartData = entries.map(e => e.isHold ? 5 : e.sc);
  const bgColors = entries.map(e =>
    e.isHold ? "rgba(251,191,36,.45)"
    : e.sc > 0 ? "rgba(52,211,153,.5)"
    : e.sc < 0 ? "rgba(248,113,113,.5)"
    : "rgba(100,116,139,.3)"
  );
  const bdColors = entries.map(e =>
    e.isHold ? "rgba(251,191,36,.85)"
    : e.sc > 0 ? "rgba(52,211,153,.85)"
    : e.sc < 0 ? "rgba(248,113,113,.85)"
    : "rgba(100,116,139,.5)"
  );

  signalChartInst = new Chart(ctx, {
    type: "bar",
    data: {
      labels: entries.map(e => e.ticker),
      datasets: [{
        data: chartData,
        backgroundColor: bgColors,
        borderColor: bdColors,
        borderWidth: 1.5,
        borderRadius: 8,
        borderSkipped: false,
        barThickness: 28,
        maxBarThickness: 34,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { left: 8, right: 8, top: 8, bottom: 0 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1a1e28", titleColor: "#e7e9ee", bodyColor: "#e7e9ee",
          borderColor: "#2e3644", borderWidth: 1, cornerRadius: 8, padding: 12,
          displayColors: false,
          callbacks: {
            title: ctx2 => entries[ctx2[0].dataIndex].ticker,
            label: ctx2 => {
              const e = entries[ctx2.dataIndex];
              const d = e.data;
              if (!d) return "尚無資料";
              const act = ACTION_LABEL[(d.action||"").toLowerCase()] || d.action || "—";
              return `${act}  ·  信心 ${d.confidence??0}%  ·  ${d.quantity??0}股  ·  $${d.current_price!=null?Number(d.current_price).toFixed(2):"—"}`;
            }
          }
        }
      },
      scales: {
        x: {
          min: -100, max: 100,
          ticks: { color: "#555e72", font: { size: 10 },
            callback: v => v===0?"0":(v>0?"買入 "+v+"%":"賣出 "+Math.abs(v)+"%"), stepSize: 25 },
          grid: { color: "rgba(35,42,54,.4)", lineWidth: 0.5 },
          title: { display: true, text: "◄  賣出 / 放空                    買入 / 回補  ►",
            color: "#555e72", font: { size: 10 } },
        },
        y: {
          ticks: { color: "#e7e9ee", font: { size: 13, weight: "600" }, padding: 8 },
          grid: { display: false },
          afterFit: scale => { scale.width = 70; },
        },
      },
    },
  });
}

/* ===== Data Table ===== */
let dashData = {};
let openTicker = null;

const STRAT_LABEL = {
  trend_following:"趨勢追蹤", mean_reversion:"均值回歸",
  momentum:"動能", volatility:"波動率", statistical_arbitrage:"統計套利",
};
const STRAT_TIP = {
  trend_following: "順勢操作：股價一路漲就看多、一路跌就看空。",
  mean_reversion: "反向操作：股價偏離平均太多就預期會拉回。",
  momentum: "看最近 1/3/6 個月的漲跌速度。",
  volatility: "看股價震盪幅度。震盪變大通常代表市場有事。",
  statistical_arbitrage: "從股價統計特性（偏度、記憶性）的異常中找機會。",
};
const SIGNAL_LABEL = {bullish:"看多", bearish:"看空", neutral:"中性"};

// Analyst display names (keyed by base agent ID prefix before the suffix)
const ANALYST_LABEL = {
  technical_analyst: "技術分析",
  nassim_taleb: "塔雷伯 (黑天鵝)",
  michael_burry: "乃大空頭",
  warren_buffett: "巴菲特",
  cathie_wood: "木頭姐",
  stanley_druckenmiller: "朱肯米勒",
  ben_graham: "葛拉漢",
  bill_ackman: "阿克曼",
  charlie_munger: "蒙格",
  peter_lynch: "彼得林區",
  phil_fisher: "費雪",
  mohnish_pabrai: "帕布萊",
  rakesh_jhunjhunwala: "印度巴菲特",
  aswath_damodaran: "達摩德蘭",
  fundamentals_analyst: "基本面分析",
  valuation_analyst: "估值分析",
  growth_analyst: "成長分析",
  sentiment_analyst: "情緒分析",
  news_sentiment_analyst: "新聞情緒",
};
const ANALYST_TIP = {
  technical_analyst: "從股價走勢、技術指標（趨勢、動能、均值回歸、波動率、統計套利）判斷方向。",
  nassim_taleb: "塔雷伯風格：專注尾部風險、反脆弱性、凸性報酬。低波動反而危險。",
  michael_burry: "大空頭風格：深度價值逆向投資，看 FCF yield 和資產負債表。",
  warren_buffett: "巴菲特風格：尋找有護城河的優質企業，長期持有。",
  cathie_wood: "木頭姐風格：破壞式創新，投資引領技術變革的公司。",
  stanley_druckenmiller: "朱肯米勒風格：宏觀趨勢投資，大押注貨幣、商品、利率。",
  ben_graham: "葛拉漢風格：安全邊際，投資被低估的公司。",
  bill_ackman: "阿克曼風格：行動派價值投資，透過策略介入釋放價值。",
  charlie_munger: "蒙格風格：護城河思維，只投資優質企業，用合理價買好公司。",
  peter_lynch: "彼得林區風格：買你懂的公司，PEG ratio 找成長股。",
  phil_fisher: "費雪風格：質化分析，看管理層能力和長期成長潛力。",
  mohnish_pabrai: "帕布萊風格：集中投資少數高確信標的，低風險高報酬。",
  rakesh_jhunjhunwala: "印度巴菲特風格：看宏觀趨勢，投資新興市場高成長產業。",
  aswath_damodaran: "達摩德蘭風格：學術派估值大師，用 DCF 和多種模型算內在價值。",
  fundamentals_analyst: "從財報數據（營收、利潤、負債比）評估公司基本面。",
  valuation_analyst: "用各種估值模型判斷股價是否合理。",
  growth_analyst: "分析成長趨勢和估值，找出成長型投資機會。",
  sentiment_analyst: "從內部人交易和市場情緒判斷多空方向。",
  news_sentiment_analyst: "從新聞報導的正負面情緒預測短期走勢。",
};

function signalBadge(sig) {
  const s = (sig||"").toLowerCase();
  return `<span class="badge ${s}">${SIGNAL_LABEL[s]||sig||"—"}</span>`;
}

function baseAgentKey(fullId) {
  // "technical_analyst_aaaaaa" → "technical_analyst"
  const parts = fullId.split("_");
  const last = parts[parts.length - 1];
  if (/^[a-z0-9]{6}$/.test(last)) return parts.slice(0, -1).join("_");
  return fullId;
}

function buildDetailHtml(ticker) {
  const d = dashData[ticker];
  if (!d || !d.payload) return `<div class="detail-panel"><div class="stat-line">無詳細資料（請重新分析一次）</div></div>`;
  const signals = (d.payload.analyst_signals||{});
  const rm = (signals["risk_management_agent_bbbbbb"]||{})[ticker]||{};
  const vol = rm.volatility_metrics||{};

  // Build analyst cards
  let analystHtml = "";
  for (const [agentId, tickerData] of Object.entries(signals)) {
    const base = baseAgentKey(agentId);
    if (base === "risk_management_agent") continue; // handled separately
    const info = (tickerData||{})[ticker]||{};
    if (!info.signal) continue;
    const label = ANALYST_LABEL[base] || base;
    const tip = ANALYST_TIP[base] || "";

    // Technical analyst has sub-strategies in reasoning
    let subHtml = "";
    if (base === "technical_analyst" && info.reasoning && typeof info.reasoning === "object") {
      subHtml = '<div class="strat-grid">';
      for (const [k, s] of Object.entries(info.reasoning)) {
        const stip = STRAT_TIP[k]||"";
        subHtml += `<div class="sk" ${stip?`data-tip="${esc(stip)}"`:""}>${STRAT_LABEL[k]||k}</div>`;
        subHtml += `<div>${signalBadge(s.signal)} <span style="color:var(--muted)">${s.confidence??0}%</span></div>`;
      }
      subHtml += "</div>";
    } else if (info.reasoning && typeof info.reasoning === "string") {
      subHtml = `<div class="reason-text">${esc(info.reasoning)}</div>`;
    }

    analystHtml += `<div class="analyst-card">
      <div class="analyst-hdr" ${tip?`data-tip="${esc(tip)}"`:""}>
        <span class="analyst-name">${esc(label)}</span>
        ${signalBadge(info.signal)} <span class="conf-sm">${info.confidence??0}%</span>
      </div>${subHtml}</div>`;
  }
  if (!analystHtml) analystHtml = `<div style="color:var(--muted)">無分析師訊號</div>`;

  // Risk management
  const riskParts = [];
  if (rm.current_price!=null) riskParts.push(`<span data-tip="最近一根日 K 的收盤價。">現價</span> $${Number(rm.current_price).toFixed(2)}`);
  if (rm.remaining_position_limit!=null) riskParts.push(`<span data-tip="風險管理允許這檔股票最多還能再投入多少錢。">剩餘倉位上限</span> $${Math.round(Number(rm.remaining_position_limit)).toLocaleString()}`);
  if (vol.annualized_volatility!=null) riskParts.push(`<span data-tip="把日波動放大成一年的尺度。20% 正常，30-40% 偏高。">年化波動</span> ${(vol.annualized_volatility*100).toFixed(1)}%`);
  if (vol.volatility_percentile!=null) riskParts.push(`<span data-tip="目前波動率跟過去比的位階。50 = 中間，80 = 比過去 80% 的時間都波動。">波動位階</span> ${vol.volatility_percentile.toFixed(0)} 分位`);

  return `<div class="detail-panel">
    <div class="dp-title" style="margin-bottom:12px">分析師觀點 (${Object.keys(signals).filter(k=>baseAgentKey(k)!=="risk_management_agent").length} 位)</div>
    <div class="analyst-grid">${analystHtml}</div>
    <div class="dp-grid" style="margin-top:16px">
      <div><div class="dp-title">風險管理</div><div class="stat-line">${riskParts.join("<br>")||"無資料"}</div></div>
      <div>${d.reasoning?`<div class="dp-title">投組經理結論</div><div class="reason-full">${esc(d.reasoning)}</div>`:""}</div>
    </div>
  </div>`;
}

function toggleDetail(ticker, event) {
  if (event && event.target.closest(".ref-btn")) return;
  const tbody = $("tableBody");
  const existing = document.getElementById("detail-row");
  const wasOpen = openTicker===ticker;
  if (existing) existing.remove();
  tbody.querySelectorAll("tr.active").forEach(r => r.classList.remove("active"));
  if (wasOpen) { openTicker=null; return; }
  const dataRows = tbody.querySelectorAll("tr.data-row");
  let targetRow = null;
  dataRows.forEach(r => { if (r.dataset.ticker===ticker) targetRow=r; });
  if (!targetRow) return;
  targetRow.classList.add("active");
  const detailRow = document.createElement("tr");
  detailRow.id = "detail-row";
  detailRow.className = "detail-row";
  detailRow.innerHTML = `<td colspan="8">${buildDetailHtml(ticker)}</td>`;
  targetRow.after(detailRow);
  openTicker = ticker;
}

function renderTable(tickers) {
  dashData = tickers;
  const tbody = $("tableBody");
  const rows = TICKERS.map(t => {
    const d = tickers[t];
    if (!d) return `<tr class="data-row" data-ticker="${t}" onclick="toggleDetail('${t}',event)">
      <td class="tk">${t}</td><td colspan="6" style="color:var(--muted)">尚無資料</td>
      <td><button class="ref-btn" onclick="event.stopPropagation();analyzeOne('${t}')">↻</button></td></tr>`;
    const confPct = d.confidence??0;
    const barColor = (d.action||"").match(/buy|cover/)?"var(--buy)":(d.action||"").match(/sell|short/)?"var(--sell)":"var(--hold)";
    return `<tr class="data-row" data-ticker="${t}" onclick="toggleDetail('${t}',event)">
      <td class="tk">${t}</td>
      <td class="price">$${d.current_price!=null?Number(d.current_price).toFixed(2):"—"}</td>
      <td>${badge(d.action)}</td>
      <td class="qty">${d.quantity??0}</td>
      <td><div class="conf-wrap"><span class="conf-bar" style="width:${Math.max(4,confPct*.8)}px;background:${barColor}"></span><span class="conf-num">${confPct}%</span></div></td>
      <td class="reasoning" title="${esc(d.reasoning||"")}">${esc(d.reasoning||"—")}</td>
      <td class="stamp">${timeAgo(d.created_at)}</td>
      <td><button class="ref-btn" onclick="event.stopPropagation();analyzeOne('${t}')">↻</button></td></tr>`;
  });
  tbody.innerHTML = rows.join("");
  openTicker = null;
}

/* ===== History Chart ===== */
let histChartInst = null;
async function renderHistoryChart() {
  try {
    const resp = await fetch("/simple/history-all?limit=30");
    const data = await resp.json();
    let hasData = false;
    const datasets = TICKERS.map(t => {
      const items = (data.tickers[t]||[]).slice().reverse();
      if (items.length) hasData = true;
      return {
        label: t,
        data: items.map(it => ({ x: new Date(it.created_at), y: signedConf(it.action, it.confidence) })),
        borderColor: TICKER_COLOR[t], backgroundColor: TICKER_COLOR[t]+"15",
        tension: .35, pointRadius: items.length<15?5:3,
        pointBackgroundColor: TICKER_COLOR[t], fill: false, borderWidth: 2,
      };
    });
    $("histEmpty").classList.toggle("hidden", hasData);
    $("historyChart").style.display = hasData?"":"none";
    if (!hasData) return;
    if (histChartInst) histChartInst.destroy();
    histChartInst = new Chart($("historyChart"), {
      type: "line", data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color:"#e7e9ee", font:{size:11}, usePointStyle:true, pointStyle:"circle", padding:16 } },
          tooltip: {
            backgroundColor:"#1a1e28", titleColor:"#e7e9ee", bodyColor:"#e7e9ee",
            borderColor:"#2e3644", borderWidth:1, cornerRadius:8, padding:12,
            callbacks: { label: ctx2 => {
              const v=ctx2.parsed.y; const dir=v>0?"買入":v<0?"賣出/放空":"觀望";
              return `${ctx2.dataset.label}: ${dir} ${Math.abs(v)}%`;
            }}
          }
        },
        scales: {
          x: { type:"time", time:{unit:"day",tooltipFormat:"yyyy-MM-dd HH:mm"},
            ticks:{color:"#555e72",font:{size:10},maxTicksLimit:10}, grid:{color:"rgba(35,42,54,.5)"} },
          y: { ticks:{color:"#555e72",font:{size:10},
            callback:v=>v===0?"0":(v>0?"買 "+v+"%":"賣 "+Math.abs(v)+"%")},
            grid:{color:"rgba(35,42,54,.35)"},
            title:{display:true,text:"◄ 看空 (賣出)     看多 (買入) ►",color:"#555e72",font:{size:10}} },
        },
      },
    });
  } catch(e) { console.error("History chart error:", e); }
}

/* ===== Data loading ===== */
async function loadDashboard() {
  try {
    const resp = await fetch("/simple/dashboard");
    const data = await resp.json();
    renderSignalChart(data.tickers);
    renderTable(data.tickers);
  } catch(e) {
    $("tableBody").innerHTML = `<tr><td colspan="8" class="empty-state" style="color:var(--sell)">載入失敗</td></tr>`;
  }
}

async function analyzeOne(ticker) {
  const btn = document.querySelector(`button[onclick="analyzeOne('${ticker}')"]`);
  if (btn) btn.innerHTML = '<span class="spin"></span>';
  try {
    const resp = await fetch("/simple/analyze", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ticker, model_name: $("model").value, model_provider:"Groq"})
    });
    if (resp.status===403) { showGate(); return; }
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({}))).detail||resp.status);
    await loadDashboard();
    await renderHistoryChart();
  } catch(e) {
    alert(`${ticker} 分析失敗: ${e.message||e}`);
    if (btn) btn.textContent = "↻";
  }
}

async function refreshAll() {
  const btn = $("refreshAll");
  btn.disabled=true; btn.textContent="分析中…";
  try {
    const results = await Promise.allSettled(TICKERS.map(t =>
      fetch("/simple/analyze", {
        method:"POST", headers: authHeaders(),
        body: JSON.stringify({ticker:t, model_name:$("model").value, model_provider:"Groq"})
      })
    ));
    // Check if any returned 403
    for (const r of results) {
      if (r.status==="fulfilled" && r.value.status===403) { showGate(); return; }
    }
    await loadDashboard();
    await renderHistoryChart();
  } finally { btn.disabled=false; btn.textContent="全部刷新"; }
}

$("refreshAll").addEventListener("click", refreshAll);

// Boot
(async () => {
  const ok = await checkAccess();
  if (ok) { loadDashboard(); renderHistoryChart(); }
})();
</script>
</body>
</html>
"""


@router.get("", response_class=HTMLResponse)
async def simple_page():
    return HTMLResponse(_HTML)
