"""
Ultra-simple UI + API for one-shot hedge-fund analysis.

Exposes:
  GET  /simple           → plain HTML page with an input + button
  POST /simple/analyze   → JSON endpoint that takes {ticker, model_name?} and
                           runs a minimal agent pipeline, returns the final
                           portfolio-manager decision + analyst signals.

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
from app.backend.models.schemas import GraphEdge, GraphNode
from app.backend.services.api_key_service import ApiKeyService
from app.backend.services.graph import (
    create_graph,
    parse_hedge_fund_response,
    run_graph_async,
)
from app.backend.services.portfolio import create_portfolio

router = APIRouter(prefix="/simple")


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

    # Date window: past 180 days
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d")

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

    return {
        "ticker": ticker,
        "window": {"start": start_date, "end": end_date},
        "model": f"{req.model_provider} / {req.model_name}",
        "decisions": decisions,
        "analyst_signals": analyst_signals,
        "current_prices": current_prices,
    }


_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Hedge Fund — 快速分析</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #181b22;
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
  .wrap { max-width: 760px; margin: 0 auto; padding: 48px 20px 80px; }
  h1 { font-size: 24px; margin: 0 0 8px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 28px; }
  .row { display: flex; gap: 12px; margin-bottom: 18px; }
  input, select, button {
    font: inherit; color: var(--text); background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
  }
  input { flex: 1; text-transform: uppercase; }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  button { background: var(--accent); border: none; color: #fff; cursor: pointer; font-weight: 600;
    padding: 12px 22px; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .hint { color: var(--muted); font-size: 12px; margin-top: -8px; margin-bottom: 18px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-top: 18px; }
  .decision { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700;
    font-size: 13px; letter-spacing: 0.5px; }
  .badge.buy { background: rgba(34,197,94,0.15); color: var(--buy); }
  .badge.sell { background: rgba(239,68,68,0.15); color: var(--sell); }
  .badge.hold { background: rgba(234,179,8,0.15); color: var(--hold); }
  .badge.short { background: rgba(239,68,68,0.15); color: var(--sell); }
  .badge.cover { background: rgba(34,197,94,0.15); color: var(--buy); }
  .price { font-size: 22px; font-weight: 600; }
  .meta { color: var(--muted); font-size: 13px; margin-top: 8px; }
  .kv { display: grid; grid-template-columns: 130px 1fr; gap: 8px 16px; margin-top: 16px;
    font-size: 14px; }
  .kv .k { color: var(--muted); }
  .reasoning { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border);
    line-height: 1.6; font-size: 14px; white-space: pre-wrap; word-break: break-word; }
  .section-title { font-size: 13px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 10px; }
  .spin { display: inline-block; width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid var(--muted); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .err { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5; padding: 14px 16px; border-radius: 8px; margin-top: 18px; font-size: 14px; }
  pre { background: #0b0d11; padding: 12px; border-radius: 8px; overflow: auto; font-size: 12px;
    color: #c7ccd7; }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI Hedge Fund — 快速分析</h1>
  <div class="sub">輸入股票代號（例如 <code>GOOG</code> / <code>AAPL</code> / <code>NVDA</code>），按分析，AI 會給你買 / 賣 / 觀望的建議和理由。</div>

  <div class="row">
    <input id="ticker" placeholder="GOOG" autofocus autocomplete="off" />
    <select id="model">
      <option value="llama-3.1-8b-instant">Llama 3.1 8B (快)</option>
      <option value="llama-3.3-70b-versatile">Llama 3.3 70B (深)</option>
      <option value="moonshotai/kimi-k2-instruct">Kimi K2</option>
      <option value="deepseek-r1-distill-llama-70b">DeepSeek R1 Distill</option>
    </select>
    <button id="go">分析</button>
  </div>
  <div class="hint">模型都在 Groq 上跑，免費額度一分鐘數次 OK。近 180 天技術分析，不會算基本面。</div>

  <div id="out"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const out = $("out");

function badge(action) {
  const a = (action || "").toLowerCase();
  const label = { buy: "買入", sell: "賣出", hold: "觀望", short: "放空", cover: "回補" }[a] || action;
  return `<span class="badge ${a}">${label}</span>`;
}

function renderResult(r) {
  const t = r.ticker;
  const dec = (r.decisions && r.decisions[t]) || {};
  const price = (r.current_prices && r.current_prices[t]);
  const signals = r.analyst_signals || {};

  let html = `<div class="card">`;
  html += `<div class="decision">${badge(dec.action)}`;
  if (dec.quantity != null) html += `<span>${dec.quantity} 股</span>`;
  if (price != null) html += `<span class="price">$${Number(price).toFixed(2)}</span>`;
  html += `</div>`;
  html += `<div class="meta">${t} · ${r.window.start} → ${r.window.end} · ${r.model}</div>`;

  if (dec.confidence != null) {
    html += `<div class="kv"><div class="k">信心度</div><div>${dec.confidence}%</div></div>`;
  }
  if (dec.reasoning) {
    html += `<div class="reasoning"><div class="section-title">Portfolio Manager 理由</div>${escapeHtml(dec.reasoning)}</div>`;
  }

  // Analyst signals
  const names = Object.keys(signals);
  if (names.length) {
    html += `<div class="reasoning"><div class="section-title">Agent 原始訊號</div>`;
    for (const n of names) {
      const sig = signals[n] && signals[n][t];
      if (!sig) continue;
      html += `<div style="margin-top:8px"><strong>${n}</strong>: `;
      html += `${badge(sig.signal)} 信心 ${sig.confidence || 0}%`;
      if (sig.reasoning) {
        const r = typeof sig.reasoning === "string" ? sig.reasoning
          : JSON.stringify(sig.reasoning);
        html += `<div style="margin-top:4px;color:var(--muted);font-size:13px">${escapeHtml(r.slice(0, 600))}${r.length > 600 ? "…" : ""}</div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
  }

  html += `</div>`;
  out.innerHTML = html;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

async function run() {
  const ticker = $("ticker").value.trim().toUpperCase();
  if (!ticker) { $("ticker").focus(); return; }
  $("go").disabled = true;
  out.innerHTML = `<div class="card"><span class="spin"></span>分析 ${ticker} 中… 約 15–40 秒</div>`;
  try {
    const resp = await fetch("/simple/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, model_name: $("model").value, model_provider: "Groq" })
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      out.innerHTML = `<div class="err">錯誤：${escapeHtml(err.detail || String(resp.status))}</div>`;
      return;
    }
    const data = await resp.json();
    renderResult(data);
  } catch (e) {
    out.innerHTML = `<div class="err">連線失敗：${escapeHtml(String(e))}</div>`;
  } finally {
    $("go").disabled = false;
  }
}

$("go").addEventListener("click", run);
$("ticker").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
</script>
</body>
</html>
"""


@router.get("", response_class=HTMLResponse)
async def simple_page():
    return HTMLResponse(_HTML)
