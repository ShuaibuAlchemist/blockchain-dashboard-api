"""
FastAPI Backend with Persistent Cache Fallback
Uses last successful Dune results when API limit is reached
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pickle
from pathlib import Path

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
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
WHALE_QUERY_ID = 5763322
INFLOW_QUERY_ID = 5781730

# Cache files for persistence
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
WHALE_CACHE_FILE = CACHE_DIR / "whale_cache.pkl"
FLOWS_CACHE_FILE = CACHE_DIR / "flows_cache.pkl"

# In-memory cache
CACHE_DURATION = 300
whale_cache = {"data": None, "timestamp": 0, "source": "none"}
flows_cache = {"data": None, "timestamp": 0, "source": "none"}

# ==========================================
# CACHE PERSISTENCE FUNCTIONS
# ==========================================

def save_cache_to_disk(cache_data, file_path):
    """Save successful Dune results to disk"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved cache to {file_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

def load_cache_from_disk(file_path):
    """Load last successful Dune results from disk"""
    try:
        if file_path.exists():
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"Loaded cache from {file_path}")
            return cache_data
        return None
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None

# ==========================================
# DATA FETCHERS WITH PERSISTENT FALLBACK
# ==========================================

def fetch_dune_whale_data():
    """Fetch whale data with persistent cache fallback"""
    now = time.time()
    
    # Return recent cache
    if whale_cache["data"] is not None and (now - whale_cache["timestamp"]) < CACHE_DURATION:
        print(f"Returning cached whale data (age: {int(now - whale_cache['timestamp'])}s, source: {whale_cache['source']})")
        return whale_cache["data"], whale_cache["source"]
    
    # Try to fetch fresh data from Dune
    try:
        if not DUNE_API_KEY:
            print("No Dune API key")
            cached = load_cache_from_disk(WHALE_CACHE_FILE)
            if cached:
                whale_cache.update({"data": cached["data"], "timestamp": now, "source": "disk_cache"})
                return cached["data"], "disk_cache"
            return pd.DataFrame(), "none"
        
        print(f"Executing Dune whale query {WHALE_QUERY_ID}")
        headers = {"x-dune-api-key": DUNE_API_KEY}
        
        response = requests.post(
            f"https://api.dune.com/api/v1/query/{WHALE_QUERY_ID}/execute",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Execute failed: {response.status_code}")
        
        run_data = response.json()
        
        if 'execution_id' not in run_data:
            raise Exception(f"No execution_id: {run_data}")
        
        execution_id = run_data['execution_id']
        
        # Poll for results
        for attempt in range(30):
            time.sleep(2)
            status_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers,
                timeout=10
            )
            status_data = status_response.json()
            status = status_data.get('state', '')
            
            if status == 'QUERY_STATE_COMPLETED':
                results_response = requests.get(
                    f"https://api.dune.com/api/v1/execution/{execution_id}/results",
                    headers=headers,
                    timeout=10
                )
                results_json = results_response.json()
                df = pd.DataFrame(results_json['result']['rows'])
                
                if not df.empty:
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                    
                    print(f"SUCCESS: Got {len(df)} whale transactions from Dune")
                    
                    # Save to both memory and disk
                    cache_data = {"data": df, "timestamp": now, "source": "dune"}
                    whale_cache.update(cache_data)
                    save_cache_to_disk(cache_data, WHALE_CACHE_FILE)
                    
                    return df, "dune"
                else:
                    print("Dune returned empty - likely hit API limit")
                    break
            
            if status == 'QUERY_STATE_FAILED':
                print("Query failed")
                break
        
    except Exception as e:
        print(f"Error fetching from Dune: {e}")
    
    # Fallback to disk cache
    print("Falling back to cached Dune results from disk")
    cached = load_cache_from_disk(WHALE_CACHE_FILE)
    if cached:
        whale_cache.update({"data": cached["data"], "timestamp": now, "source": "disk_cache"})
        return cached["data"], "disk_cache"
    
    return pd.DataFrame(), "none"

def fetch_dune_exchange_flows():
    """Fetch exchange flows with persistent cache fallback"""
    now = time.time()
    
    # Return recent cache
    if flows_cache["data"] is not None and (now - flows_cache["timestamp"]) < CACHE_DURATION:
        print(f"Returning cached flow data (age: {int(now - flows_cache['timestamp'])}s, source: {flows_cache['source']})")
        return flows_cache["data"], flows_cache["source"]
    
    # Try to fetch fresh data from Dune
    try:
        if not DUNE_API_KEY:
            cached = load_cache_from_disk(FLOWS_CACHE_FILE)
            if cached:
                flows_cache.update({"data": cached["data"], "timestamp": now, "source": "disk_cache"})
                return cached["data"], "disk_cache"
            return pd.DataFrame(), "none"
        
        print(f"Executing Dune exchange flows query {INFLOW_QUERY_ID}")
        headers = {"x-dune-api-key": DUNE_API_KEY}
        
        response = requests.post(
            f"https://api.dune.com/api/v1/query/{INFLOW_QUERY_ID}/execute",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Execute failed: {response.status_code}")
        
        run_data = response.json()
        
        if 'execution_id' not in run_data:
            raise Exception(f"No execution_id: {run_data}")
        
        execution_id = run_data['execution_id']
        
        # Poll for results
        for attempt in range(30):
            time.sleep(2)
            status_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/status",
                headers=headers,
                timeout=10
            )
            status_data = status_response.json()
            status = status_data.get('state', '')
            
            if status == 'QUERY_STATE_COMPLETED':
                results_response = requests.get(
                    f"https://api.dune.com/api/v1/execution/{execution_id}/results",
                    headers=headers,
                    timeout=10
                )
                results_json = results_response.json()
                df = pd.DataFrame(results_json['result']['rows'])
                
                if not df.empty:
                    df['inflow'] = pd.to_numeric(df['inflow'], errors='coerce').fillna(0)
                    df['outflow'] = pd.to_numeric(df['outflow'], errors='coerce').fillna(0)
                    df['week_start'] = pd.to_datetime(df['week_start']).dt.tz_localize(None)
                    
                    print(f"SUCCESS: Got {len(df)} exchange flow records from Dune")
                    
                    # Save to both memory and disk
                    cache_data = {"data": df, "timestamp": now, "source": "dune"}
                    flows_cache.update(cache_data)
                    save_cache_to_disk(cache_data, FLOWS_CACHE_FILE)
                    
                    return df, "dune"
                else:
                    print("Dune returned empty - likely hit API limit")
                    break
            
            if status == 'QUERY_STATE_FAILED':
                break
        
    except Exception as e:
        print(f"Error fetching from Dune: {e}")
    
    # Fallback to disk cache
    print("Falling back to cached Dune results from disk")
    cached = load_cache_from_disk(FLOWS_CACHE_FILE)
    if cached:
        flows_cache.update({"data": cached["data"], "timestamp": now, "source": "disk_cache"})
        return cached["data"], "disk_cache"
    
    return pd.DataFrame(), "none"

def fetch_live_prices():
    """Fetch live prices from CoinGecko"""
    try:
        response = requests.get(
            f"{COINGECKO_BASE}/simple/price",
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
            return response.json()
        return {}
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return {}

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Blockchain Risk & Transparency Dashboard API",
        "timestamp": datetime.now().isoformat(),
        "data_source": {
            "whale_data": whale_cache.get("source", "none"),
            "flow_data": flows_cache.get("source", "none")
        },
        "info": "Using cached Dune data when API limit is reached"
    }

@app.get("/api/whale-transfers")
def get_whale_transfers():
    try:
        df, source = fetch_dune_whale_data()
        
        if df.empty:
            return {
                "transactions": [],
                "count": 0,
                "data_source": source,
                "message": "No data available - Dune limit may be reached"
            }
        
        transactions = []
        for _, row in df.head(100).iterrows():
            transactions.append({
                "tx_hash": row.get('tx_hash', ''),
                "timestamp": row.get('timestamp').isoformat() if pd.notna(row.get('timestamp')) else None,
                "from_address": row.get('from_address', ''),
                "to_address": row.get('to_address', ''),
                "amount": float(row.get('amount', 0)),
                "token": row.get('token', ''),
            })
        
        return {
            "transactions": transactions,
            "count": len(transactions),
            "data_source": source
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exchange-flows")
def get_exchange_flows():
    try:
        df, source = fetch_dune_exchange_flows()
        
        if df.empty:
            return {
                "flows": [],
                "count": 0,
                "data_source": source,
                "message": "No data available - Dune limit may be reached"
            }
        
        flows = []
        for _, row in df.iterrows():
            flows.append({
                "exchange": row.get('exchange', 'Unknown'),
                "token": row.get('token', 'ETH'),
                "week_start": row.get('week_start').isoformat() if pd.notna(row.get('week_start')) else None,
                "inflow": float(row.get('inflow', 0)),
                "outflow": float(row.get('outflow', 0)),
                "net_flow": float(row.get('outflow', 0)) - float(row.get('inflow', 0))
            })
        
        return {
            "flows": flows,
            "count": len(flows),
            "data_source": source
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-overview")
def get_market_overview():
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
