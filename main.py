"""
Complete FastAPI Backend for Blockchain Risk & Transparency Dashboard
with intelligent caching and fallback handling.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ==============================================================
# Load environment variables
# ==============================================================
load_dotenv()

app = FastAPI(title="Blockchain Risk Dashboard API")

# ==============================================================
# Enable CORS
# ==============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================
# Configuration
# ==============================================================
DUNE_API_KEY = os.getenv("DUNE_KEY")
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
WHALE_QUERY_ID = 5763322
INFLOW_QUERY_ID = 5781730

# Cache duration (seconds)
CACHE_DURATION = 300
whale_cache = {"data": None, "timestamp": 0}
flows_cache = {"data": None, "timestamp": 0}
prices_cache = {"data": None, "timestamp": 0}

# Contract to token mapping
CONTRACT_TO_TOKEN = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC"
}


# ==============================================================
# DATA COLLECTION FUNCTIONS WITH CACHING
# ==============================================================

def fetch_dune_whale_data():
    """Fetch whale transfers from Dune with caching"""
    now = time.time()

    if whale_cache["data"] is not None and (now - whale_cache["timestamp"]) < CACHE_DURATION:
        print(f"Returning cached whale data (age: {int(now - whale_cache['timestamp'])}s)")
        return whale_cache["data"]

    try:
        print(f"Executing fresh Dune whale query {WHALE_QUERY_ID}")
        headers = {"x-dune-api-key": DUNE_API_KEY}
        run_query_url = f"https://api.dune.com/api/v1/query/{WHALE_QUERY_ID}/execute"
        response = requests.post(run_query_url, headers=headers)
        run_data = response.json()

        if "execution_id" not in run_data:
            print(f"Failed to execute query: {run_data}")
            return pd.DataFrame()

        execution_id = run_data["execution_id"]
        print(f"Execution ID: {execution_id}")

        status = ""
        max_attempts = 30
        attempt = 0

        while status not in ["QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED"] and attempt < max_attempts:
            time.sleep(2)
            status_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers
            )
            status_data = status_response.json()
            status = status_data.get("state", "")
            attempt += 1

        if status == "QUERY_STATE_COMPLETED":
            results_url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
            results_response = requests.get(results_url, headers=headers)
            results_json = results_response.json()
            df = pd.DataFrame(results_json["result"]["rows"])

            print(f"Got {len(df)} whale transactions")

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            whale_cache["data"] = df
            whale_cache["timestamp"] = now

            return df
        else:
            print(f"Query failed or timed out. Status: {status}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching whale data: {e}")
        return pd.DataFrame()


def fetch_dune_exchange_flows():
    """Fetch exchange inflow/outflow from Dune with caching"""
    now = time.time()

    if flows_cache["data"] is not None and (now - flows_cache["timestamp"]) < CACHE_DURATION:
        print(f"Returning cached flows data (age: {int(now - flows_cache['timestamp'])}s)")
        return flows_cache["data"]

    try:
        print(f"Executing fresh Dune exchange flows query {INFLOW_QUERY_ID}")
        headers = {"x-dune-api-key": DUNE_API_KEY}
        run_query_url = f"https://api.dune.com/api/v1/query/{INFLOW_QUERY_ID}/execute"
        response = requests.post(run_query_url, headers=headers)
        run_data = response.json()

        if "execution_id" not in run_data:
            print(f"Failed to execute query: {run_data}")
            return pd.DataFrame()

        execution_id = run_data["execution_id"]
        status = ""
        max_attempts = 30
        attempt = 0

        while status not in ["QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED"] and attempt < max_attempts:
            time.sleep(2)
            status_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers
            )
            status_data = status_response.json()
            status = status_data.get("state", "")
            attempt += 1

        if status == "QUERY_STATE_COMPLETED":
            results_url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
            results_response = requests.get(results_url, headers=headers)
            results_json = results_response.json()
            df = pd.DataFrame(results_json["result"]["rows"])

            # Clean data
            df["inflow"] = pd.to_numeric(df["inflow"], errors="coerce").fillna(0)
            df["outflow"] = pd.to_numeric(df["outflow"], errors="coerce").fillna(0)
            df["week_start"] = pd.to_datetime(df["week_start"]).dt.tz_localize(None)
            df["contract_address"] = df["contract_address"].str.lower()
            df["token"] = df["contract_address"].map(CONTRACT_TO_TOKEN)

            print(f"Got {len(df)} exchange flow records")

            flows_cache["data"] = df
            flows_cache["timestamp"] = now

            return df
        else:
            print(f"Query failed or timed out. Status: {status}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching exchange flows: {e}")
        return pd.DataFrame()


def fetch_live_prices():
    """Fetch current prices from CoinGecko with caching"""
    now = time.time()
    if prices_cache["data"] is not None and (now - prices_cache["timestamp"]) < 60:
        print(f"Returning cached prices (age: {int(now - prices_cache['timestamp'])}s)")
        return prices_cache["data"]

    try:
        print("Fetching fresh prices from CoinGecko")
        assets = ["ethereum", "bitcoin", "tether", "usd-coin", "wrapped-bitcoin"]
        url = f"{COINGECKO_BASE}/simple/price"
        response = requests.get(url, params={
            "ids": ",".join(assets),
            "vs_currencies": "usd",
            "include_24h_vol": "true",
            "include_24h_change": "true",
            "include_market_cap": "true"
        })

        if response.status_code == 200:
            data = response.json()
            print(f"Got prices for {len(data)} assets")
            prices_cache["data"] = data
            prices_cache["timestamp"] = now
            return data
        return {}
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return {}


def fetch_price_history(days=7):
    """Fetch historical ETH price data"""
    try:
        url = f"{COINGECKO_BASE}/coins/ethereum/market_chart"
        response = requests.get(url, params={
            "vs_currency": "usd",
            "days": days
        })

        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching price history: {e}")
        return pd.DataFrame()


# ==============================================================
# API ENDPOINTS
# ==============================================================

@app.get("/")
def read_root():
    """Health check"""
    return {
        "status": "active",
        "message": "Blockchain Risk & Transparency Dashboard API",
        "timestamp": datetime.now().isoformat(),
        "cache_status": {
            "whale_data_age": int(time.time() - whale_cache["timestamp"]) if whale_cache["data"] else None,
            "flows_data_age": int(time.time() - flows_cache["timestamp"]) if flows_cache["data"] else None,
            "prices_age": int(time.time() - prices_cache["timestamp"]) if prices_cache["data"] else None
        }
    }


@app.get("/api/market-overview")
def get_market_overview():
    """Get current market overview"""
    try:
        prices = fetch_live_prices()
        if not prices:
            raise HTTPException(status_code=500, detail="Failed to fetch prices")

        eth_data = prices.get("ethereum", {})
        return {
            "eth_price": eth_data.get("usd", 0),
            "eth_volume_24h": eth_data.get("usd_24h_vol", 0),
            "market_cap": eth_data.get("usd_market_cap", 0),
            "price_change_24h": eth_data.get("usd_24h_change", 0),
            "timestamp": int(datetime.now().timestamp())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/whale-transfers")
def get_whale_transfers():
    """Get recent whale transactions"""
    try:
        df = fetch_dune_whale_data()
        if df.empty:
            return {"transactions": [], "count": 0}

        transactions = []
        for _, row in df.head(100).iterrows():
            transactions.append({
                "tx_hash": row.get("tx_hash", ""),
                "timestamp": row.get("timestamp").isoformat() if pd.notna(row.get("timestamp")) else None,
                "from_address": row.get("from_address", ""),
                "to_address": row.get("to_address", ""),
                "amount": float(row.get("amount", 0)),
                "token": row.get("token", "")
            })

        return {"transactions": transactions, "count": len(transactions)}
    except Exception as e:
        print(f"Error in whale-transfers endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/exchange-flows")
def get_exchange_flows():
    """Get exchange inflow/outflow data"""
    try:
        df = fetch_dune_exchange_flows()
        if df.empty:
            return {"flows": [], "count": 0}

        summary = df.groupby(["exchange", "token", "week_start"]).agg({
            "inflow": "sum",
            "outflow": "sum"
        }).reset_index()

        flows = []
        for _, row in summary.iterrows():
            if pd.notna(row["token"]):
                flows.append({
                    "exchange": row["exchange"],
                    "token": row["token"],
                    "week_start": row["week_start"].isoformat(),
                    "inflow": float(row["inflow"]),
                    "outflow": float(row["outflow"]),
                    "net_flow": float(row["outflow"] - row["inflow"])
                })

        return {"flows": flows, "count": len(flows)}
    except Exception as e:
        print(f"Error in exchange-flows endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/flow-summary")
def get_flow_summary():
    """Get aggregated flow summary"""
    try:
        df = fetch_dune_exchange_flows()
        if df.empty:
            return {"total_inflow": 0, "total_outflow": 0, "net_flow": 0, "sentiment": "neutral"}

        total_inflow = df["inflow"].sum()
        total_outflow = df["outflow"].sum()
        net_flow = total_outflow - total_inflow

        sentiment = "bullish" if net_flow > 0 else "bearish" if net_flow < 0 else "neutral"
        return {
            "total_inflow": float(total_inflow),
            "total_outflow": float(total_outflow),
            "net_flow": float(net_flow),
            "sentiment": sentiment
        }
    except Exception as e:
        print(f"Error in flow-summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price-history")
def get_price_history_endpoint(days: int = 7):
    """Get ETH price history"""
    try:
        df = fetch_price_history(days)
        if df.empty:
            return {"prices": [], "count": 0}

        prices = [{"timestamp": row["timestamp"].isoformat(), "price": float(row["price"])} for _, row in df.iterrows()]
        return {"prices": prices, "count": len(prices)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh-data")
def refresh_all_data():
    """Force refresh all cached data"""
    try:
        whale_cache["timestamp"] = 0
        flows_cache["timestamp"] = 0
        prices_cache["timestamp"] = 0

        whale_df = fetch_dune_whale_data()
        flows_df = fetch_dune_exchange_flows()
        prices = fetch_live_prices()

        return {
            "status": "success",
            "whale_transactions": len(whale_df),
            "exchange_flows": len(flows_df),
            "prices_fetched": len(prices),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

            whale_cache["data"] = df
            whale_cache["timestamp"] = now
            return df

        print(f"Query failed or timed out. Status: {status}")
        return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching whale data: {e}")
        return pd.DataFrame()


def fetch_dune_exchange_flows():
    """Fetch exchange inflow/outflow from Dune with caching"""
    now = time.time()
    if flows_cache["data"] is not None and (now - flows_cache["timestamp"]) < CACHE_DURATION:
        print(f"Returning cached flows data (age: {int(now - flows_cache['timestamp'])}s)")
        return flows_cache["data"]

    try:
        print(f"Executing fresh Dune exchange flows query {INFLOW_QUERY_ID}")
        headers = {"x-dune-api-key": DUNE_API_KEY}
        run_query_url = f"https://api.dune.com/api/v1/query/{INFLOW_QUERY_ID}/execute"
        response = requests.post(run_query_url, headers=headers)
        run_data = response.json()

        # Handle datapoint limit or quota issue
        if "error" in run_data and "datapoint limit" in run_data["error"].lower():
            print("⚠️ Dune datapoint limit reached. Using cached flow data if available.")
            return flows_cache["data"] if flows_cache["data"] is not None else pd.DataFrame()

        if 'execution_id' not in run_data:
            print(f"Failed to execute query: {run_data}")
            return pd.DataFrame()

        execution_id = run_data['execution_id']
        print(f"Execution ID: {execution_id}")

        # Poll for completion
        status = ''
        attempt = 0
        while status not in ['QUERY_STATE_COMPLETED', 'QUERY_STATE_FAILED'] and attempt < 30:
            time.sleep(2)
            status_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers
            )
            status_data = status_response.json()
            status = status_data.get('state', '')
            attempt += 1

        if status == 'QUERY_STATE_COMPLETED':
            results_url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
            results_response = requests.get(results_url, headers=headers)
            results_json = results_response.json()
            df = pd.DataFrame(results_json['result']['rows'])
            print(f"Got {len(df)} exchange flow records")

            # Clean data
            df['inflow'] = pd.to_numeric(df.get('inflow', 0), errors='coerce').fillna(0)
            df['outflow'] = pd.to_numeric(df.get('outflow', 0), errors='coerce').fillna(0)
            df['week_start'] = pd.to_datetime(df['week_start']).dt.tz_localize(None)
            df['contract_address'] = df['contract_address'].str.lower()
            df['token'] = df['contract_address'].map(CONTRACT_TO_TOKEN)

            flows_cache["data"] = df
            flows_cache["timestamp"] = now
            return df

        print(f"Query failed or timed out. Status: {status}")
        return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching exchange flows: {e}")
        return pd.DataFrame()


def fetch_live_prices():
    """Fetch current prices from CoinGecko with caching"""
    now = time.time()
    if prices_cache["data"] is not None and (now - prices_cache["timestamp"]) < 60:
        print(f"Returning cached prices (age: {int(now - prices_cache['timestamp'])}s)")
        return prices_cache["data"]

    try:
        print("Fetching fresh prices from CoinGecko")
        assets = ["ethereum", "bitcoin", "tether", "usd-coin", "wrapped-bitcoin"]
        url = f"{COINGECKO_BASE}/simple/price"
        response = requests.get(url, params={
            "ids": ",".join(assets),
            "vs_currencies": "usd",
            "include_24h_vol": "true",
            "include_24h_change": "true",
            "include_market_cap": "true"
        })

        if response.status_code == 200:
            data = response.json()
            prices_cache["data"] = data
            prices_cache["timestamp"] = now
            return data

        return {}
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return {}


def fetch_price_history(days=7):
    """Fetch historical price data"""
    try:
        url = f"{COINGECKO_BASE}/coins/ethereum/market_chart"
        response = requests.get(url, params={
            "vs_currency": "usd",
            "days": days
        })

        if response.status_code == 200:
            data = response.json()
            prices = data.get('prices', [])
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching price history: {e}")
        return pd.DataFrame()


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    """Health check"""
    return {
        "status": "active",
        "message": "Blockchain Risk & Transparency Dashboard API",
        "timestamp": datetime.now().isoformat(),
        "cache_status": {
            "whale_data_age": int(time.time() - whale_cache["timestamp"]) if whale_cache["data"] is not None else None,
            "flows_data_age": int(time.time() - flows_cache["timestamp"]) if flows_cache["data"] is not None else None,
            "prices_age": int(time.time() - prices_cache["timestamp"]) if prices_cache["data"] is not None else None
        }
    }


@app.get("/api/market-overview")
def get_market_overview():
    """Get current market overview"""
    try:
        prices = fetch_live_prices()
        if not prices:
            raise HTTPException(status_code=500, detail="Failed to fetch prices")

        eth_data = prices.get('ethereum', {})
        return {
            "eth_price": eth_data.get('usd', 0),
            "eth_volume_24h": eth_data.get('usd_24h_vol', 0),
            "market_cap": eth_data.get('usd_market_cap', 0),
            "price_change_24h": eth_data.get('usd_24h_change', 0),
            "timestamp": int(datetime.now().timestamp())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/whale-transfers")
def get_whale_transfers():
    """Get recent whale transactions"""
    try:
        df = fetch_dune_whale_data()
        if df.empty:
            return {"transactions": [], "count": 0}

        transactions = []
        for _, row in df.head(100).iterrows():
            transactions.append({
                "tx_hash": row.get('tx_hash', ''),
                "timestamp": row.get('timestamp').isoformat() if pd.notna(row.get('timestamp')) else None,
                "from_address": row.get('from_address', ''),
                "to_address": row.get('to_address', ''),
                "amount": float(row.get('amount', 0)),
                "token": row.get('token', '')
            })

        return {"transactions": transactions, "count": len(transactions)}

    except Exception as e:
        print(f"Error in whale-transfers endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/exchange-flows")
def get_exchange_flows():
    """Get exchange inflow/outflow data"""
    try:
        df = fetch_dune_exchange_flows()
        if df.empty:
            return {"flows": [], "count": 0}

        summary = df.groupby(['exchange', 'token', 'week_start']).agg({
            'inflow': 'sum',
            'outflow': 'sum'
        }).reset_index()

        flows = []
        for _, row in summary.iterrows():
            if pd.notna(row['token']):
                flows.append({
                    "exchange": row['exchange'],
                    "token": row['token'],
                    "week_start": row['week_start'].isoformat(),
                    "inflow": float(row['inflow']),
                    "outflow": float(row['outflow']),
                    "net_flow": float(row['outflow'] - row['inflow'])
                })

        return {"flows": flows, "count": len(flows)}

    except Exception as e:
        print(f"Error in exchange-flows endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/flow-summary")
def get_flow_summary():
    """Get aggregated flow summary"""
    try:
        df = fetch_dune_exchange_flows()
        if df.empty:
            return {"total_inflow": 0, "total_outflow": 0, "net_flow": 0, "sentiment": "neutral"}

        total_inflow = df['inflow'].sum()
        total_outflow = df['outflow'].sum()
        net_flow = total_outflow - total_inflow

        sentiment = "bullish" if net_flow > 0 else "bearish" if net_flow < 0 else "neutral"

        return {
            "total_inflow": float(total_inflow),
            "total_outflow": float(total_outflow),
            "net_flow": float(net_flow),
            "sentiment": sentiment
        }

    except Exception as e:
        print(f"Error in flow-summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price-history")
def get_price_history_endpoint(days: int = 7):
    """Get ETH price history"""
    try:
        df = fetch_price_history(days)
        if df.empty:
            return {"prices": [], "count": 0}

        prices = [{"timestamp": row['timestamp'].isoformat(), "price": float(row['price'])} for _, row in df.iterrows()]
        return {"prices": prices, "count": len(prices)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh-data")
def refresh_all_data():
    """Force refresh all cached data"""
    try:
        whale_cache["timestamp"] = 0
        flows_cache["timestamp"] = 0
        prices_cache["timestamp"] = 0

        whale_df = fetch_dune_whale_data()
        flows_df = fetch_dune_exchange_flows()
        prices = fetch_live_prices()

        return {
            "status": "success",
            "whale_transactions": len(whale_df),
            "exchange_flows": len(flows_df),
            "prices_fetched": len(prices),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# RUN LOCALLY
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
            df = pd.DataFrame(results_json["result"]["rows"])
            print(f"✅ Got {len(df)} whale transactions")

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            # Cache the result
            whale_cache["data"] = df
            whale_cache["timestamp"] = now
            return df
        else:
            print(f"⚠️ Query failed or timed out. Status: {status}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error fetching whale data: {e}")
        return pd.DataFrame()


def fetch_dune_exchange_flows():
    """Fetch exchange inflow/outflow from Dune with caching"""
    now = time.time()

    if flows_cache["data"] is not None and (now - flows_cache["timestamp"]) < CACHE_DURATION:
        print(f"✅ Returning cached flows data (age: {int(now - flows_cache['timestamp'])}s)")
        return flows_cache["data"]

    try:
        print(f"🔄 Executing fresh Dune exchange flows query {INFLOW_QUERY_ID}")
        headers = {"x-dune-api-key": DUNE_API_KEY}

        run_query_url = f"https://api.dune.com/api/v1/query/{INFLOW_QUERY_ID}/execute"
        response = requests.post(run_query_url, headers=headers)
        run_data = response.json()

        if "execution_id" not in run_data:
            print(f"❌ Failed to execute query: {run_data}")
            return pd.DataFrame()

        execution_id = run_data["execution_id"]

        status = ""
        for _ in range(30):
            time.sleep(2)
            status_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers,
            )
            status_data = status_response.json()
            status = status_data.get("state", "")
            if status in ["QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED"]:
                break

        if status == "QUERY_STATE_COMPLETED":
            results_url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
            results_response = requests.get(results_url, headers=headers)
            results_json = results_response.json()
            df = pd.DataFrame(results_json["result"]["rows"])

            # Clean and normalize
            df["inflow"] = pd.to_numeric(df.get("inflow", 0), errors="coerce").fillna(0)
            df["outflow"] = pd.to_numeric(df.get("outflow", 0), errors="coerce").fillna(0)
            df["week_start"] = pd.to_datetime(df["week_start"]).dt.tz_localize(None)
            df["contract_address"] = df["contract_address"].astype(str).str.lower()
            df["token"] = df["contract_address"].map(CONTRACT_TO_TOKEN)

            print(f"✅ Got {len(df)} exchange flow records")

            flows_cache["data"] = df
            flows_cache["timestamp"] = now
            return df
        else:
            print(f"⚠️ Query failed or timed out. Status: {status}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error fetching exchange flows: {e}")
        return pd.DataFrame()


def fetch_live_prices():
    """Fetch current prices from CoinGecko with caching"""
    now = time.time()
    if prices_cache["data"] is not None and (now - prices_cache["timestamp"]) < 60:
        print(f"✅ Returning cached prices (age: {int(now - prices_cache['timestamp'])}s)")
        return prices_cache["data"]

    try:
        print("🔄 Fetching fresh prices from CoinGecko")
        assets = ["ethereum", "bitcoin", "tether", "usd-coin", "wrapped-bitcoin"]
        response = requests.get(
            f"{COINGECKO_BASE}/simple/price",
            params={
                "ids": ",".join(assets),
                "vs_currencies": "usd",
                "include_24h_vol": "true",
                "include_24h_change": "true",
                "include_market_cap": "true",
            },
        )

        if response.status_code == 200:
            data = response.json()
            prices_cache["data"] = data
            prices_cache["timestamp"] = now
            print(f"✅ Got prices for {len(data)} assets")
            return data

        return {}

    except Exception as e:
        print(f"❌ Error fetching prices: {e}")
        return {}


def fetch_price_history(days=7):
    """Fetch historical ETH price data"""
    try:
        url = f"{COINGECKO_BASE}/coins/ethereum/market_chart"
        response = requests.get(url, params={"vs_currency": "usd", "days": days})
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get("prices", []), columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching price history: {e}")
        return pd.DataFrame()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    """Health check"""
    return {
        "status": "active",
        "message": "Blockchain Risk & Transparency Dashboard API",
        "timestamp": datetime.now().isoformat(),
        "cache_status": {
            "whale_data_age": int(time.time() - whale_cache["timestamp"]) if whale_cache["data"] else None,
            "flows_data_age": int(time.time() - flows_cache["timestamp"]) if flows_cache["data"] else None,
            "prices_age": int(time.time() - prices_cache["timestamp"]) if prices_cache["data"] else None,
        },
    }


@app.get("/api/market-overview")
def get_market_overview():
    """Get current market overview"""
    prices = fetch_live_prices()
    if not prices:
        raise HTTPException(status_code=500, detail="Failed to fetch prices")

    eth_data = prices.get("ethereum", {})
    return {
        "eth_price": eth_data.get("usd", 0),
        "eth_volume_24h": eth_data.get("usd_24h_vol", 0),
        "market_cap": eth_data.get("usd_market_cap", 0),
        "price_change_24h": eth_data.get("usd_24h_change", 0),
        "timestamp": int(datetime.now().timestamp()),
    }


@app.get("/api/whale-transfers")
def get_whale_transfers():
    """Get recent whale transactions"""
    df = fetch_dune_whale_data()
    if df.empty:
        return {"transactions": [], "count": 0}

    transactions = [
        {
            "tx_hash": row.get("tx_hash", ""),
            "timestamp": row["timestamp"].isoformat() if pd.notna(row.get("timestamp")) else None,
            "from_address": row.get("from_address", ""),
            "to_address": row.get("to_address", ""),
            "amount": float(row.get("amount", 0)),
            "token": row.get("token", ""),
        }
        for _, row in df.head(100).iterrows()
    ]

    return {"transactions": transactions, "count": len(transactions)}


@app.get("/api/exchange-flows")
def get_exchange_flows():
    """Get exchange inflow/outflow data"""
    df = fetch_dune_exchange_flows()
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


@app.get("/api/flow-summary")
def get_flow_summary():
    """Get aggregated flow summary"""
    df = fetch_dune_exchange_flows()
    if df.empty:
        return {"total_inflow": 0, "total_outflow": 0, "net_flow": 0, "sentiment": "neutral"}

    total_inflow = df["inflow"].sum()
    total_outflow = df["outflow"].sum()
    net_flow = total_outflow - total_inflow
    sentiment = "bullish" if net_flow > 0 else "bearish" if net_flow < 0 else "neutral"

    return {
        "total_inflow": float(total_inflow),
        "total_outflow": float(total_outflow),
        "net_flow": float(net_flow),
        "sentiment": sentiment,
    }


@app.get("/api/price-history")
def get_price_history_endpoint(days: int = 7):
    """Get ETH price history"""
    df = fetch_price_history(days)
    if df.empty:
        return {"prices": [], "count": 0}

    prices = [
        {"timestamp": row["timestamp"].isoformat(), "price": float(row["price"])}
        for _, row in df.iterrows()
    ]

    return {"prices": prices, "count": len(prices)}


@app.post("/api/refresh-data")
def refresh_all_data():
    """Force refresh all cached data"""
    try:
        whale_cache["timestamp"] = 0
        flows_cache["timestamp"] = 0
        prices_cache["timestamp"] = 0

        whale_df = fetch_dune_whale_data()
        flows_df = fetch_dune_exchange_flows()
        prices = fetch_live_prices()

        return {
            "status": "success",
            "whale_transactions": len(whale_df),
            "exchange_flows": len(flows_df),
            "prices_fetched": len(prices),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Run locally
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)            return pd.DataFrame()

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
