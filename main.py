"""
Enhanced FastAPI Backend for Blockchain Risk & Transparency Dashboard
- Intelligent caching
- Graceful Dune API fallback
- Fully JSON-safe responses
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
app = FastAPI(title="Blockchain Risk Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DUNE_API_KEY = os.getenv("DUNE_KEY")
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Dune query IDs
WHALE_QUERY_ID = 5763322
INFLOW_QUERY_ID = 5781730

# Cache dictionaries
CACHE_TTL = 300  # 5 minutes
CACHE = {
    "whales": {"data": None, "timestamp": 0},
    "flows": {"data": None, "timestamp": 0},
    "prices": {"data": None, "timestamp": 0},
}

CONTRACT_TO_TOKEN = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def cache_is_fresh(key: str) -> bool:
    return (
        CACHE[key]["data"] is not None
        and (time.time() - CACHE[key]["timestamp"]) < CACHE_TTL
    )

def safe_json(data):
    """Ensure any NaN or non-serializable types are safely converted"""
    return (
        data.replace({float("nan"): None})
        if isinstance(data, pd.DataFrame)
        else data
    )

# ==========================================
# DUNE API HELPERS
# ==========================================

def execute_dune_query(query_id: int):
    """Executes a Dune query and waits for completion"""
    headers = {"x-dune-api-key": DUNE_API_KEY}

    # Step 1: Start execution
    exec_res = requests.post(
        f"https://api.dune.com/api/v1/query/{query_id}/execute", headers=headers
    )
    run_data = exec_res.json()
    execution_id = run_data.get("execution_id")

    if not execution_id:
        print(f"❌ Failed to start query: {run_data}")
        return pd.DataFrame()

    # Step 2: Poll for completion
    for _ in range(30):
        time.sleep(2)
        status_res = requests.get(
            f"https://api.dune.com/api/v1/execution/{execution_id}/status",
            headers=headers,
        )
        state = status_res.json().get("state", "")
        if state == "QUERY_STATE_COMPLETED":
            break
        elif state == "QUERY_STATE_FAILED":
            print("❌ Query failed.")
            return pd.DataFrame()

    # Step 3: Fetch results
    results = requests.get(
        f"https://api.dune.com/api/v1/execution/{execution_id}/results",
        headers=headers,
    ).json()

    rows = results.get("result", {}).get("rows", [])
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

# ==========================================
# DATA FETCHERS WITH CACHING & FALLBACKS
# ==========================================

def get_whale_data():
    """Fetch whale transfers (cached or from Dune)"""
    if cache_is_fresh("whales"):
        print("✅ Returning cached whale data")
        return CACHE["whales"]["data"]

    print("🔄 Fetching whale data from Dune...")
    df = execute_dune_query(WHALE_QUERY_ID)

    if df.empty:
        print("⚠️ No whale data found")
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    CACHE["whales"] = {"data": df, "timestamp": time.time()}
    return df


def get_exchange_flows():
    """Fetch exchange flows (cached or from Dune)"""
    if cache_is_fresh("flows"):
        print("✅ Returning cached flow data")
        return CACHE["flows"]["data"]

    print("🔄 Fetching exchange flows from Dune...")
    df = execute_dune_query(INFLOW_QUERY_ID)

    if df.empty:
        print("⚠️ No exchange flow data found")
        return pd.DataFrame()

    df["inflow"] = pd.to_numeric(df.get("inflow", 0), errors="coerce").fillna(0)
    df["outflow"] = pd.to_numeric(df.get("outflow", 0), errors="coerce").fillna(0)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.tz_localize(None)
    df["contract_address"] = df.get("contract_address", "").astype(str).str.lower()
    df["token"] = df["contract_address"].map(CONTRACT_TO_TOKEN)

    CACHE["flows"] = {"data": df, "timestamp": time.time()}
    return df


def get_live_prices():
    """Fetch live prices with caching"""
    if cache_is_fresh("prices"):
        return CACHE["prices"]["data"]

    try:
        url = f"{COINGECKO_BASE}/simple/price"
        assets = ["ethereum", "bitcoin", "tether", "usd-coin", "wrapped-bitcoin"]
        response = requests.get(
            url,
            params={
                "ids": ",".join(assets),
                "vs_currencies": "usd",
                "include_24h_vol": "true",
                "include_24h_change": "true",
                "include_market_cap": "true",
            },
        )

        data = response.json() if response.status_code == 200 else {}
        CACHE["prices"] = {"data": data, "timestamp": time.time()}
        return data

    except Exception as e:
        print(f"Error fetching prices: {e}")
        return {}

# ==========================================
# ROUTES
# ==========================================

@app.get("/")
def root():
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "cache_status": {
            k: int(time.time() - v["timestamp"]) if v["data"] else None
            for k, v in CACHE.items()
        },
    }


@app.get("/api/whale-transfers")
def whale_transfers():
    df = get_whale_data()
    if df.empty:
        return {"transactions": [], "count": 0}

    txs = [
        {
            "tx_hash": row.get("tx_hash", ""),
            "timestamp": row.get("timestamp").isoformat()
            if pd.notna(row.get("timestamp"))
            else None,
            "from_address": row.get("from_address", ""),
            "to_address": row.get("to_address", ""),
            "amount": float(row.get("amount", 0)),
            "token": row.get("token", ""),
        }
        for _, row in df.head(100).iterrows()
    ]

    return {"transactions": txs, "count": len(txs)}


@app.get("/api/exchange-flows")
def exchange_flows():
    df = get_exchange_flows()
    if df.empty:
        return {"flows": [], "count": 0}

    summary = (
        df.groupby(["exchange", "token", "week_start"])
        .agg({"inflow": "sum", "outflow": "sum"})
        .reset_index()
    )

    flows = [
        {
            "exchange": row["exchange"],
            "token": row["token"],
            "week_start": row["week_start"].isoformat(),
            "inflow": float(row["inflow"]),
            "outflow": float(row["outflow"]),
            "net_flow": float(row["outflow"] - row["inflow"]),
        }
        for _, row in summary.iterrows()
        if pd.notna(row["token"])
    ]

    return {"flows": flows, "count": len(flows)}


@app.get("/api/market-overview")
def market_overview():
    prices = get_live_prices()
    eth = prices.get("ethereum", {})
    if not eth:
        raise HTTPException(status_code=500, detail="Failed to fetch ETH data")

    return {
        "eth_price": eth.get("usd", 0),
        "eth_volume_24h": eth.get("usd_24h_vol", 0),
        "market_cap": eth.get("usd_market_cap", 0),
        "price_change_24h": eth.get("usd_24h_change", 0),
        "timestamp": int(datetime.now().timestamp()),
    }


@app.post("/api/refresh-data")
def refresh_data():
    """Force-refresh all data"""
    for key in CACHE.keys():
        CACHE[key]["timestamp"] = 0
    return {
        "status": "refreshed",
        "timestamp": datetime.now().isoformat(),
    }

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
