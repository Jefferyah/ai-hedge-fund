from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import os
from datetime import datetime, timedelta, timezone

from app.backend.routes import api_router
from app.backend.database.connection import engine, SessionLocal
from app.backend.database.models import Base
from app.backend.repositories.api_key_repository import ApiKeyRepository
from app.backend.services.ollama_service import ollama_service

# API key providers that can be seeded from environment variables.
# SQLite state is ephemeral on Zeabur container rebuilds, so we re-seed
# from env vars on startup — Zeabur env vars are the source of truth.
_SEEDABLE_API_KEYS = [
    "GROQ_API_KEY",
    "FINANCIAL_DATASETS_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MOONSHOT_API_KEY",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

# Initialize database tables (this is safe to run multiple times)
Base.metadata.create_all(bind=engine)

# Configure CORS
# Default allows local dev. Set BACKEND_CORS_ORIGINS as a comma-separated list
# of allowed origins (e.g. "https://my-frontend.zeabur.app,https://foo.com")
# to enable additional origins in deployed environments.
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [
    o.strip()
    for o in os.getenv("BACKEND_CORS_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)


def _seed_api_keys_from_env() -> dict:
    """Seed the api_keys table from environment variables.

    Zeabur env vars are the source of truth. On every container start we
    upsert any present env-var keys into the DB so every code path (DB
    lookup or env fallback) finds the same value. This avoids the "keys
    disappeared after rebuild" problem without needing a persistent volume.
    """
    seeded = []
    skipped = []
    db = SessionLocal()
    try:
        repo = ApiKeyRepository(db)
        for provider in _SEEDABLE_API_KEYS:
            value = os.getenv(provider)
            if not value:
                skipped.append(provider)
                continue
            repo.create_or_update_api_key(
                provider=provider,
                key_value=value,
                description=f"Seeded from env var at startup",
                is_active=True,
            )
            seeded.append(provider)
    finally:
        db.close()
    return {"seeded": seeded, "skipped": skipped}


@app.get("/healthz")
def healthz():
    """Liveness + key-presence check. Returns which API keys are loaded
    (both env var and DB) so you can tell at a glance whether the container
    has what it needs to run an analysis."""
    db = SessionLocal()
    try:
        db_keys = {k.provider for k in ApiKeyRepository(db).get_all_api_keys(include_inactive=False)}
    finally:
        db.close()
    return {
        "ok": True,
        "api_keys": {
            p: {
                "env": bool(os.getenv(p)),
                "db": p in db_keys,
            }
            for p in _SEEDABLE_API_KEYS
        },
    }


@app.on_event("startup")
async def startup_event():
    """Seed API keys from env vars, then check Ollama availability."""
    try:
        result = _seed_api_keys_from_env()
        if result["seeded"]:
            logger.info(f"Seeded API keys from env: {', '.join(result['seeded'])}")
        else:
            logger.info("No API keys present in env vars to seed")
    except Exception as e:
        logger.warning(f"Failed to seed API keys from env: {e}")

    try:
        logger.info("Checking Ollama availability...")
        status = await ollama_service.check_ollama_status()
        
        if status["installed"]:
            if status["running"]:
                logger.info(f"✓ Ollama is installed and running at {status['server_url']}")
                if status["available_models"]:
                    logger.info(f"✓ Available models: {', '.join(status['available_models'])}")
                else:
                    logger.info("ℹ No models are currently downloaded")
            else:
                logger.info("ℹ Ollama is installed but not running")
                logger.info("ℹ You can start it from the Settings page or manually with 'ollama serve'")
        else:
            logger.info("ℹ Ollama is not installed. Install it to use local models.")
            logger.info("ℹ Visit https://ollama.com to download and install Ollama")
            
    except Exception as e:
        logger.warning(f"Could not check Ollama status: {e}")
        logger.info("ℹ Ollama integration is available if you install it later")

    # Launch the daily auto-refresh background loop.
    asyncio.create_task(_daily_refresh_loop())


# ---------------------------------------------------------------------------
# Daily auto-refresh of dashboard watchlist
# ---------------------------------------------------------------------------
# Runs at ~21:05 UTC every day (= ~17:05 ET, shortly after US market close).
# Configurable via DAILY_REFRESH_HOUR env var (default 21). Set to -1 to
# disable entirely.

_DAILY_HOUR = int(os.getenv("DAILY_REFRESH_HOUR", "21"))
_DAILY_MINUTE = 5


def _seconds_until_next_run() -> float:
    """Return seconds until the next scheduled run, always > 0."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=_DAILY_HOUR, minute=_DAILY_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def _run_all_analyses() -> dict:
    """Run /simple/analyze for every watchlist ticker (sequentially to be
    gentle with the Groq free tier rate limit)."""
    from app.backend.routes.simple_ui import WATCHLIST, SimpleAnalyzeRequest, simple_analyze
    from app.backend.database import get_db

    results = {}
    db = SessionLocal()
    try:
        for ticker in WATCHLIST:
            try:
                req = SimpleAnalyzeRequest(
                    ticker=ticker,
                    model_name="llama-3.1-8b-instant",
                    model_provider="Groq",
                )
                r = await simple_analyze(req, db)
                results[ticker] = r.get("decisions", {}).get(ticker, {}).get("action", "?")
                # Small pause between tickers to respect Groq free-tier rate limits
                await asyncio.sleep(5)
            except Exception as e:
                results[ticker] = f"error: {e}"
                logger.warning(f"Auto-refresh failed for {ticker}: {e}")
    finally:
        db.close()
    return results


async def _daily_refresh_loop():
    """Background loop: sleep until next scheduled time, run analyses, repeat."""
    if _DAILY_HOUR < 0:
        logger.info("Daily auto-refresh disabled (DAILY_REFRESH_HOUR=-1)")
        return

    logger.info(
        f"Daily auto-refresh enabled: will run at {_DAILY_HOUR:02d}:{_DAILY_MINUTE:02d} UTC every day"
    )
    while True:
        wait = _seconds_until_next_run()
        logger.info(f"Next auto-refresh in {wait/3600:.1f}h")
        await asyncio.sleep(wait)
        logger.info("Starting daily auto-refresh…")
        try:
            results = await _run_all_analyses()
            logger.info(f"Daily auto-refresh done: {results}")
        except Exception as e:
            logger.error(f"Daily auto-refresh crashed: {e}")
