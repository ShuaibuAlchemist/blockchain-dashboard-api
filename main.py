"""
Blockchain Risk & Transparency Dashboard API (with CoinGecko Pro)
Author: Muhammad Shuaibu
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, time, requests, pandas as pd, random, pickle
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# ==========================================
# Load environment and setup
# ==========================================
load_dotenv()
app = FastAPI(title="Blockchain Risk Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DUNE_API_KEY = os.getenv("DUNE_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE = "https://pro-api.coingecko.com/api/v3"

WHALE_QUERY_ID = 5763322
INFLOW_QUERY_ID = 5781730

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
WHALE_CACHE_FILE = CACHE_DIR / "whale_cache.pkl"
FLOWS_CACHE_FILE = CACHE_DIR / "flows_cache.pkl"

CACHE_DURATION = 300  # 5 minutes
whale_cache = {"data": None, "timestamp": 0, "source": "none"}
flows_cache = {"data": None, "timestamp": 0, "source": "none"}

# ==========================================
# MOCK DATA GENERATORS
# ==========================================
def generate_mock_whale_data():
    now = datetime.now()
    tokens = ['ETH', 'USDT', 'USDC']
    data = []
    for i in range(50):
        token = random.choice(tokens)
        amount = random.uniform(1000, 50000) if token == 'ETH' else random.uniform(1_000_000, 10_000_000)
        data.append({
            "tx_hash": f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
            "timestamp": now - timedelta(hours=i*2),
            "from_address": f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            "to_address": f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            "amount": amount,
            "token": token
        })
    return pd.DataFrame(data)

def generate_mock_flow_data():
    now = datetime.now()
    exchanges = ['Binance', 'Coinbase', 'Kraken']
    tokens = ['ETH', 'USDT', 'USDC']
    flows = []
    for week_offset in range(12):
        week_start = now - timedelta(weeks=week_offset)
        for exchange in exchanges:
            for token in tokens:
                inflow = random.uniform(10_000, 100_000)
                outflow = random.uniform(10_000, 100_000)
                flows.append({
                    "exchange": exchange,
                    "token": token,
                    "week_start": week_start,
                    "inflow": inflow,
                    "outflow": outflow
                })
    return pd.DataFrame(flows)

# ==========================================
# CACHE HELPERS
# ==========================================
def save_cache(cache_data, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Failed to save cache: {e}")

def load_cache(file_path):
    try:
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Failed to load cache: {e}")
    return None

def initialize_cache():
    if not WHALE_CACHE_FILE.exists():
        df = generate_mock_whale_data()
        save_cache({"data": df, "timestamp": time.time(), "source": "mock"}, WHALE_CACHE_FILE)
    if not FLOWS_CACHE_FILE.exists():
        df = generate_mock_flow_data()
        save_cache({"data": df, "timestamp": time.time(), "source": "mock"}, FLOWS_CACHE_FILE)

initialize_cache()

# ==========================================
# DATA FETCHING FUNCTIONS
# ==========================================
def fetch_live_prices():
    """Fetch live prices from CoinGecko Pro"""
    try:
        headers = {"x-cg-pro-api-key": COINGECKO_API_KEY}
        response = requests.get(
            f"{COINGECKO_BASE}/simple/price",
            headers=headers,
            params={
                "ids": "ethereum,bitcoin,tether,usd-coin",
                "vs_currencies": "usd",
                "include_24h_vol": "true",
                "include_24h_change": "true",
                "include_market_cap": "true"
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print("✅ Live price data fetched from CoinGecko Pro.")
            return data
        else:
            print(f"❌ CoinGecko Pro request failed: {response.status_code} {response.text}")
            return {}
    except Exception as e:
        print(f"⚠ Error fetching prices: {e}")
        return {}

def fetch_dune_data(query_id, cache_file, cache):
    """Reusable Dune data fetcher"""
    now = time.time()
    if cache["data"] is not None and (now - cache["timestamp"]) < CACHE_DURATION:
        return cache["data"], cache["source"]

    try:
        if not DUNE_API_KEY:
            cached = load_cache(cache_file)
            if cached:
                cache.update({"data": cached["data"], "timestamp": now, "source": cached["source"]})
                return cached["data"], cached["source"]
            return pd.DataFrame(), "none"

        headers = {"x-dune-api-key": DUNE_API_KEY}
        exec_res = requests.post(
            f"https://api.dune.com/api/v1/query/{query_id}/execute",
            headers=headers, timeout=10
        )

        if exec_res.status_code != 200:
            raise Exception(f"Dune execution failed: {exec_res.status_code}")

        execution_id = exec_res.json().get("execution_id")
        if not execution_id:
            raise Exception("No execution_id found in Dune response")

        for _ in range(30):
            time.sleep(2)
            status_res = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers, timeout=10
            )
            if status_res.json().get("state") == "QUERY_STATE_COMPLETED":
                results = requests.get(
                    f"https://api.dune.com/api/v1/execution/{execution_id}/results",
                    headers=headers, timeout=10
                ).json()
                df = pd.DataFrame(results['result']['rows'])
                if not df.empty:
                    cache_data = {"data": df, "timestamp": now, "source": "dune"}
                    cache.update(cache_data)
                    save_cache(cache_data, cache_file)
                    return df, "dune"
        raise Exception("Dune query timed out or failed.")
    except Exception as e:
        print(f"Dune fetch error: {e}")
        cached = load_cache(cache_file)
        if cached:
            cache.update({"data": cached["data"], "timestamp": now, "source": cached["source"]})
            return cached["data"], cached["source"]
        return pd.DataFrame(), "error"

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def root():
    return {
        "status": "active",
        "message": "Blockchain Risk Dashboard API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/market-overview")
def get_market_overview():
    try:
        prices = fetch_live_prices()
        if not prices or "ethereum" not in prices:
            raise HTTPException(status_code=500, detail="Failed to fetch market data from CoinGecko Pro.")

        eth = prices["ethereum"]
        return {
            "eth_price": eth.get("usd", 0),
            "eth_volume_24h": eth.get("usd_24h_vol", 0),
            "market_cap": eth.get("usd_market_cap", 0),
            "price_change_24h": eth.get("usd_24h_change", 0),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/whale-transfers")
def whale_transfers():
    df, src = fetch_dune_data(WHALE_QUERY_ID, WHALE_CACHE_FILE, whale_cache)
    if df.empty:
        return {"transactions": [], "count": 0, "data_source": src}
    txs = df.head(100).to_dict(orient="records")
    return {"transactions": txs, "count": len(txs), "data_source": src}

@app.get("/api/exchange-flows")
def exchange_flows():
    df, src = fetch_dune_data(INFLOW_QUERY_ID, FLOWS_CACHE_FILE, flows_cache)
    if df.empty:
        return {"flows": [], "count": 0, "data_source": src}
    df["net_flow"] = df["outflow"] - df["inflow"]
    flows = df.to_dict(orient="records")
    return {"flows": flows, "count": len(flows), "data_source": src}

# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
