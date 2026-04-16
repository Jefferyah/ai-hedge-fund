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
<title>AI Hedge Fund Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root {
    --bg: #0f1115; --panel: #181b22; --panel-2: #1f232d;
    --border: #2a2f3a; --text: #e7e9ee; --muted: #8a92a3;
    --accent: #4f8cff; --buy: #22c55e; --sell: #ef4444; --hold: #eab308;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 20px 60px; }

  /* Header */
  header { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; }
  h1 { font-size: 20px; font-weight: 700; flex: 1; white-space: nowrap; }
  select, button { font: inherit; font-size: 13px; color: var(--text); background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; cursor: pointer; }
  select:focus { outline: none; border-color: var(--accent); }
  .primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  .primary:disabled { opacity: .5; cursor: not-allowed; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 22px; line-height: 1.6; }

  /* Tooltip */
  [data-tip] { border-bottom: 1px dotted rgba(138,146,163,.4); cursor: help; position: relative; }
  [data-tip]:hover { border-bottom-color: var(--accent); }
  [data-tip]::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 8px);
    left: 50%; transform: translateX(-50%); background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px;
    font-size: 12px; line-height: 1.5; color: var(--text); width: 250px; white-space: normal;
    box-shadow: 0 8px 24px rgba(0,0,0,.5); opacity: 0; pointer-events: none;
    transition: opacity .12s; z-index: 100; }
  [data-tip]:hover::after { opacity: 1; }

  /* Section panels */
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px; margin-bottom: 16px; }
  .panel-title { font-size: 13px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .8px; margin-bottom: 14px; }

  /* Signal bar chart */
  #signalChart { height: 200px; }

  /* Data table */
  .tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
  .tbl th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 10px;
    border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase;
    letter-spacing: .6px; }
  .tbl td { padding: 12px 10px; border-bottom: 1px solid rgba(42,47,58,.4);
    vertical-align: middle; }
  .tbl tr:last-child td { border-bottom: none; }
  .tbl tr:hover td { background: rgba(79,140,255,.03); }
  .tbl .tk { font-weight: 700; font-size: 15px; letter-spacing: .5px; }
  .tbl .price { font-weight: 600; }
  .tbl .conf-bar { display: inline-block; height: 6px; border-radius: 3px; min-width: 4px; }
  .tbl .reasoning { color: var(--muted); font-size: 11px; max-width: 200px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tbl .stamp { color: var(--muted); font-size: 11px; }
  .ref-btn { background: transparent; border: 1px solid var(--border); color: var(--muted);
    width: 28px; height: 28px; padding: 0; border-radius: 6px; font-size: 13px; cursor: pointer; }
  .ref-btn:hover { color: var(--accent); border-color: var(--accent); }

  /* Badges */
  .badge { display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-weight: 700; font-size: 11px; letter-spacing: .4px; white-space: nowrap; }
  .badge.buy, .badge.cover { background: rgba(34,197,94,.15); color: var(--buy); }
  .badge.sell, .badge.short { background: rgba(239,68,68,.15); color: var(--sell); }
  .badge.hold { background: rgba(234,179,8,.15); color: var(--hold); }
  .badge.bullish { background: rgba(34,197,94,.15); color: var(--buy); }
  .badge.bearish { background: rgba(239,68,68,.15); color: var(--sell); }
  .badge.neutral { background: rgba(138,146,163,.15); color: var(--muted); }

  /* Clickable rows */
  .tbl tbody tr { cursor: pointer; }
  .tbl tbody tr.active td { background: rgba(79,140,255,.06); }

  /* Detail panel (inserted as a full-width row below the clicked row) */
  .detail-row td { padding: 0 !important; border-bottom: 1px solid var(--border) !important; }
  .detail-panel { padding: 16px 14px 18px; background: var(--panel-2); border-radius: 0 0 8px 8px; }
  .detail-panel .dp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .detail-panel .dp-grid { grid-template-columns: 1fr; } }
  .detail-panel .dp-section { }
  .detail-panel .dp-title { font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .7px; margin-bottom: 8px; }
  .detail-panel .strat-grid { display: grid; grid-template-columns: max-content auto;
    gap: 5px 14px; font-size: 12px; }
  .detail-panel .strat-grid .sk { color: var(--muted); }
  .detail-panel .stat-line { color: var(--muted); font-size: 12px; line-height: 1.8; }
  .detail-panel .reason-full { font-size: 12px; line-height: 1.6; margin-top: 4px; color: var(--text); }

  /* History chart */
  #historyChart { height: 260px; }

  /* Spinner */
  .spin { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid var(--border); border-top-color: var(--accent);
    animation: spin .8s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .empty-state { text-align: center; color: var(--muted); padding: 32px 0; font-size: 14px; }
  .empty-state button { margin-top: 10px; }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>AI Hedge Fund Dashboard</h1>
    <select id="model">
      <option value="llama-3.1-8b-instant">Llama 3.1 8B (快)</option>
      <option value="llama-3.3-70b-versatile">Llama 3.3 70B (深)</option>
      <option value="moonshotai/kimi-k2-instruct">Kimi K2</option>
      <option value="deepseek-r1-distill-llama-70b">DeepSeek R1 Distill</option>
    </select>
    <button class="primary" id="refreshAll">全部刷新</button>
  </header>
  <div class="sub">
    每天美東 17:05 自動更新。打開頁面顯示上次的結果，按「全部刷新」立刻重跑。
    <span data-tip="AI 假設你有 $100,000 虛擬資金，根據 180 天技術分析（股價走勢，不看財報）決定要買還是賣、買幾股。不是投資建議。">滑鼠移到有虛線的字上面可以看說明。</span>
  </div>

  <!-- Signal Overview -->
  <div class="panel">
    <div class="panel-title">
      <span data-tip="橫向柱狀圖：往右 = 建議買入（綠色），往左 = 建議賣出或放空（紅色），長度 = AI 的信心程度。">訊號總覽</span>
      — 一眼看出該買還是該賣
    </div>
    <div style="position:relative"><canvas id="signalChart"></canvas></div>
  </div>

  <!-- Data Table -->
  <div class="panel">
    <div class="panel-title">明細</div>
    <table class="tbl">
      <thead>
        <tr>
          <th>代號</th>
          <th data-tip="最近一根日 K 的收盤價。">現價</th>
          <th data-tip="AI 的交易建議。買入 = 預期上漲；賣出/放空 = 預期下跌；觀望 = 訊號不明確。">動作</th>
          <th data-tip="基於 $100,000 虛擬資金，風險管理計算出的建議部位大小（不是真錢）。">股數</th>
          <th data-tip="AI 對這個決策的把握程度。100% = 非常確定，20% = 只是稍微偏向這個方向。">信心</th>
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
      <span data-tip="Y 軸 = 方向性信心度：正值代表看多（買入），負值代表看空（賣出/放空）。數值越大代表越篤定。五檔股票各一條線。">歷史走勢</span>
      — 五檔方向性信心度隨時間變化
    </div>
    <div style="position:relative"><canvas id="historyChart"></canvas></div>
    <div id="histEmpty" class="empty-state" style="display:none">
      還沒有歷史資料。按「全部刷新」跑第一次分析後就會出現。
    </div>
  </div>

</div>

<script>
/* ===== Constants ===== */
const TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"];
const TICKER_COLOR = {
  AAPL:  "#4f8cff",
  GOOGL: "#22c55e",
  MSFT:  "#a78bfa",
  NVDA:  "#22d3ee",
  TSLA:  "#ef4444",
};
const ACTION_LABEL = {buy:"買入",sell:"賣出",hold:"觀望",short:"放空",cover:"回補"};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

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

/* sign: buy/cover → +1, sell/short → -1, hold → 0 */
function signedConf(action, conf) {
  const a = (action||"").toLowerCase();
  if (a==="buy"||a==="cover") return (conf||0);
  if (a==="sell"||a==="short") return -(conf||0);
  return 0;
}

/* ===== Signal Overview bar chart ===== */
let signalChartInst = null;
function renderSignalChart(tickers) {
  const ctx = $("signalChart");
  if (signalChartInst) signalChartInst.destroy();

  // Sort by signed confidence so strongest signals on top
  const entries = TICKERS.map(t => {
    const d = tickers[t];
    return { ticker: t, sc: d ? signedConf(d.action, d.confidence) : 0, data: d };
  }).sort((a,b) => a.sc - b.sc); // most negative (sell) first

  const labels = entries.map(e => e.ticker);
  const values = entries.map(e => e.sc);
  const colors = values.map(v => v > 0 ? "rgba(34,197,94,.7)" : v < 0 ? "rgba(239,68,68,.7)" : "rgba(234,179,8,.5)");
  const borderColors = values.map(v => v > 0 ? "rgba(34,197,94,1)" : v < 0 ? "rgba(239,68,68,1)" : "rgba(234,179,8,.8)");

  signalChartInst = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
        barThickness: 28,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx2 => {
              const e = entries[ctx2.dataIndex];
              const d = e.data;
              if (!d) return "尚無資料";
              const act = ACTION_LABEL[(d.action||"").toLowerCase()] || d.action || "—";
              return `${act}  |  信心 ${d.confidence??0}%  |  ${d.quantity??0}股  |  $${d.current_price!=null?Number(d.current_price).toFixed(2):"—"}`;
            }
          }
        }
      },
      scales: {
        x: {
          min: -100, max: 100,
          ticks: {
            color: "#8a92a3", font: { size: 11 },
            callback: v => v === 0 ? "0" : (v > 0 ? "買入 "+v+"%" : "賣出 "+Math.abs(v)+"%"),
            stepSize: 25,
          },
          grid: { color: "rgba(42,47,58,.4)" },
          title: { display: true, text: "◄ 賣出 / 放空          買入 / 回補 ►", color: "#8a92a3", font: { size: 11 } },
        },
        y: {
          ticks: { color: "#e7e9ee", font: { size: 14, weight: "bold" } },
          grid: { display: false },
        },
      },
    },
  });
}

/* ===== Data Table ===== */
let dashData = {}; // global store so detail panel can read payloads
let openTicker = null;

const STRAT_LABEL = {
  trend_following: "趨勢追蹤", mean_reversion: "均值回歸",
  momentum: "動能", volatility: "波動率", statistical_arbitrage: "統計套利",
};
const STRAT_TIP = {
  trend_following: "順勢操作：股價一路漲就看多、一路跌就看空。",
  mean_reversion: "反向操作：股價偏離平均太多就預期會拉回。",
  momentum: "看最近 1/3/6 個月的漲跌速度。",
  volatility: "看股價震盪幅度。震盪變大通常代表市場有事。",
  statistical_arbitrage: "從股價統計特性（偏度、記憶性）的異常中找機會。",
};
const SIGNAL_LABEL = {bullish:"看多", bearish:"看空", neutral:"中性"};

function signalBadge(sig) {
  const s = (sig||"").toLowerCase();
  const label = SIGNAL_LABEL[s] || sig || "—";
  return `<span class="badge ${s}">${label}</span>`;
}

function buildDetailHtml(ticker) {
  const d = dashData[ticker];
  if (!d || !d.payload) return `<div class="detail-panel"><div class="stat-line">無詳細資料（請重新分析一次）</div></div>`;

  const payload = d.payload;
  const signals = payload.analyst_signals || {};
  const ta = (signals["technical_analyst_aaaaaa"] || {})[ticker] || {};
  const rm = (signals["risk_management_agent_bbbbbb"] || {})[ticker] || {};
  const strats = (ta.reasoning && typeof ta.reasoning === "object") ? ta.reasoning : {};
  const vol = rm.volatility_metrics || {};

  // Technical analysis sub-strategies
  let stratHtml = "";
  for (const [k, s] of Object.entries(strats)) {
    const tip = STRAT_TIP[k] || "";
    stratHtml += `<div class="sk" ${tip ? `data-tip="${esc(tip)}"` : ""}>${STRAT_LABEL[k]||k}</div>`;
    stratHtml += `<div>${signalBadge(s.signal)} <span style="color:var(--muted)">${s.confidence??0}%</span></div>`;
  }
  if (!stratHtml) stratHtml = `<div class="sk" style="grid-column:1/-1">無子策略資料</div>`;

  // Risk management stats
  const riskParts = [];
  if (rm.current_price != null)
    riskParts.push(`<span data-tip="最近一根日 K 的收盤價。">現價</span> $${Number(rm.current_price).toFixed(2)}`);
  if (rm.remaining_position_limit != null)
    riskParts.push(`<span data-tip="風險管理允許這檔股票最多還能再投入多少錢，用來避免單檔押太重。">剩餘倉位上限</span> $${Math.round(Number(rm.remaining_position_limit)).toLocaleString()}`);
  if (vol.annualized_volatility != null)
    riskParts.push(`<span data-tip="把日波動放大成一年的尺度。20% 左右正常，30-40% 偏高。">年化波動</span> ${(vol.annualized_volatility*100).toFixed(1)}%`);
  if (vol.volatility_percentile != null)
    riskParts.push(`<span data-tip="目前波動率跟過去比的位階。50 = 中間、80 = 比過去 80% 的時間都波動。">波動位階</span> ${vol.volatility_percentile.toFixed(0)} 分位`);

  // Overall technical signal
  const taSignal = ta.signal ? `${signalBadge(ta.signal)} 信心 ${ta.confidence??0}%` : "";

  return `<div class="detail-panel">
    <div class="dp-grid">
      <div class="dp-section">
        <div class="dp-title">技術分析 ${taSignal}</div>
        <div class="strat-grid">${stratHtml}</div>
      </div>
      <div class="dp-section">
        <div class="dp-title">風險管理</div>
        <div class="stat-line">${riskParts.join("<br>")}</div>
        ${d.reasoning ? `<div class="dp-title" style="margin-top:14px">Portfolio Manager 完整理由</div><div class="reason-full">${esc(d.reasoning)}</div>` : ""}
      </div>
    </div>
  </div>`;
}

function toggleDetail(ticker, event) {
  // Don't trigger when clicking the refresh button
  if (event && event.target.closest(".ref-btn")) return;

  const tbody = $("tableBody");
  // Remove existing detail row
  const existing = document.getElementById("detail-row");
  const wasOpen = openTicker === ticker;
  if (existing) existing.remove();
  // Remove active highlight
  tbody.querySelectorAll("tr.active").forEach(r => r.classList.remove("active"));

  if (wasOpen) { openTicker = null; return; }

  // Find the clicked data row and insert detail row after it
  const dataRows = tbody.querySelectorAll("tr.data-row");
  let targetRow = null;
  dataRows.forEach(r => { if (r.dataset.ticker === ticker) targetRow = r; });
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
    if (!d) return `<tr class="data-row" data-ticker="${t}" onclick="toggleDetail('${t}', event)">
      <td class="tk">${t}</td><td colspan="6" style="color:var(--muted)">尚無資料 — 按右邊 ↻ 分析</td>
      <td><button class="ref-btn" onclick="event.stopPropagation();analyzeOne('${t}')">↻</button></td>
    </tr>`;
    const confPct = d.confidence ?? 0;
    const barColor = (d.action||"").match(/buy|cover/) ? "var(--buy)"
      : (d.action||"").match(/sell|short/) ? "var(--sell)" : "var(--hold)";
    return `<tr class="data-row" data-ticker="${t}" onclick="toggleDetail('${t}', event)">
      <td class="tk">${t}</td>
      <td class="price">$${d.current_price!=null?Number(d.current_price).toFixed(2):"—"}</td>
      <td>${badge(d.action)}</td>
      <td>${d.quantity??0}</td>
      <td>
        <span class="conf-bar" style="width:${Math.max(4,confPct*.6)}px;background:${barColor}"></span>
        ${confPct}%
      </td>
      <td class="reasoning" title="${esc(d.reasoning||"")}">${esc(d.reasoning||"—")}</td>
      <td class="stamp">${timeAgo(d.created_at)}</td>
      <td><button class="ref-btn" onclick="event.stopPropagation();analyzeOne('${t}')">↻</button></td>
    </tr>`;
  });
  tbody.innerHTML = rows.join("");
  openTicker = null; // reset open state after re-render
}

/* ===== History Trend Chart ===== */
let histChartInst = null;
async function renderHistoryChart() {
  try {
    const resp = await fetch("/simple/history-all?limit=30");
    const data = await resp.json();

    let hasData = false;
    const datasets = TICKERS.map(t => {
      const items = (data.tickers[t] || []).slice().reverse(); // chronological
      if (items.length) hasData = true;
      return {
        label: t,
        data: items.map(it => ({
          x: new Date(it.created_at),
          y: signedConf(it.action, it.confidence),
        })),
        borderColor: TICKER_COLOR[t],
        backgroundColor: TICKER_COLOR[t] + "18",
        tension: .35,
        pointRadius: items.length < 15 ? 5 : 3,
        pointBackgroundColor: TICKER_COLOR[t],
        fill: false,
        borderWidth: 2,
      };
    });

    $("histEmpty").style.display = hasData ? "none" : "block";
    $("historyChart").style.display = hasData ? "" : "none";
    if (!hasData) return;

    if (histChartInst) histChartInst.destroy();
    histChartInst = new Chart($("historyChart"), {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#e7e9ee", font: { size: 12 }, usePointStyle: true, pointStyle: "circle" } },
          tooltip: {
            callbacks: {
              label: ctx2 => {
                const v = ctx2.parsed.y;
                const dir = v > 0 ? "買入" : v < 0 ? "賣出/放空" : "觀望";
                return `${ctx2.dataset.label}: ${dir} ${Math.abs(v)}%`;
              }
            }
          }
        },
        scales: {
          x: {
            type: "time",
            time: { unit: "day", tooltipFormat: "yyyy-MM-dd HH:mm" },
            ticks: { color: "#8a92a3", font: { size: 10 }, maxTicksLimit: 10 },
            grid: { color: "rgba(42,47,58,.4)" },
          },
          y: {
            ticks: {
              color: "#8a92a3", font: { size: 11 },
              callback: v => v === 0 ? "0" : (v > 0 ? "買 "+v+"%" : "賣 "+Math.abs(v)+"%"),
            },
            grid: { color: "rgba(42,47,58,.3)" },
            title: { display: true, text: "◄ 看空 (賣出)     看多 (買入) ►", color: "#8a92a3", font: { size: 11 } },
          },
        },
      },
    });
  } catch (e) {
    console.error("History chart error:", e);
  }
}

/* ===== Data loading ===== */
async function loadDashboard() {
  try {
    const resp = await fetch("/simple/dashboard");
    const data = await resp.json();
    renderSignalChart(data.tickers);
    renderTable(data.tickers);
  } catch (e) {
    $("tableBody").innerHTML = `<tr><td colspan="8" class="empty-state" style="color:var(--sell)">載入失敗：${esc(String(e))}</td></tr>`;
  }
}

async function analyzeOne(ticker) {
  const btn = document.querySelector(`button[onclick="analyzeOne('${ticker}')"]`);
  if (btn) btn.innerHTML = '<span class="spin"></span>';
  try {
    const resp = await fetch("/simple/analyze", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ ticker, model_name: $("model").value, model_provider: "Groq" })
    });
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({}))).detail || resp.status);
    await loadDashboard();
    await renderHistoryChart();
  } catch (e) {
    alert(`${ticker} 分析失敗: ${e.message||e}`);
    if (btn) btn.textContent = "↻";
  }
}

async function refreshAll() {
  const btn = $("refreshAll");
  btn.disabled = true; btn.textContent = "分析中…";
  try {
    await Promise.allSettled(TICKERS.map(t =>
      fetch("/simple/analyze", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ ticker: t, model_name: $("model").value, model_provider: "Groq" })
      })
    ));
    await loadDashboard();
    await renderHistoryChart();
  } finally {
    btn.disabled = false; btn.textContent = "全部刷新";
  }
}

$("refreshAll").addEventListener("click", refreshAll);
loadDashboard();
renderHistoryChart();
</script>
</body>
</html>
"""


@router.get("", response_class=HTMLResponse)
async def simple_page():
    return HTMLResponse(_HTML)
