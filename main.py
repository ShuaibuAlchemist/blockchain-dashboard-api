"""
Complete FastAPI Backend for Blockchain Risk & Transparency Dashboard
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dune_client.client import DuneClient
from typing import Dict, List
import sqlite3

# Load environment variables
load_dotenv()

app = FastAPI(title="Blockchain Risk Dashboard API")

# Enable CORS for v0.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DUNE_API_KEY = os.getenv("DUNE_KEY")
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
WHALE_QUERY_ID = 5763322
INFLOW_QUERY_ID = 5781730
DB_PATH = "data/blockchain.db"

dune_client = DuneClient(DUNE_API_KEY)

# Contract to token mapping
CONTRACT_TO_TOKEN = {
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'USDC',
    '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2': 'WETH',
    '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599': 'WBTC'
}

ASSET_TO_TOKEN = {
    'usd-coin': 'USDC',
    'tether': 'USDT',
    'ethereum': 'ETH',
    'wrapped-bitcoin': 'WBTC',
}

# ==========================================
# DATA COLLECTION FUNCTIONS
# ==========================================

def fetch_dune_whale_data():
    """Fetch whale transfers from Dune"""
    try:
        result = dune_client.get_latest_result(WHALE_QUERY_ID)
        df = pd.DataFrame(result.result.rows)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        return df
    except Exception as e:
        print(f"Error fetching whale data: {e}")
        return pd.DataFrame()

def fetch_dune_exchange_flows():
    """Fetch exchange inflow/outflow from Dune"""
    try:
        headers = {"x-dune-api-key": DUNE_API_KEY}
        run_query_url = f"https://api.dune.com/api/v1/query/{INFLOW_QUERY_ID}/execute"
        response = requests.post(run_query_url, headers=headers)
        run_data = response.json()
        execution_id = run_data['execution_id']

        # Poll for completion
        status = ''
        max_attempts = 30
        attempt = 0
        
        while status not in ['QUERY_STATE_COMPLETED', 'QUERY_STATE_FAILED'] and attempt < max_attempts:
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

            # Clean data
            df['inflow'] = pd.to_numeric(df['inflow'], errors='coerce').fillna(0)
            df['outflow'] = pd.to_numeric(df['outflow'], errors='coerce').fillna(0)
            df['week_start'] = pd.to_datetime(df['week_start']).dt.tz_localize(None)
            df['contract_address'] = df['contract_address'].str.lower()
            df['token'] = df['contract_address'].map(CONTRACT_TO_TOKEN)
            
            return df
        else:
            print(f"Query failed or timed out. Status: {status}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching exchange flows: {e}")
        return pd.DataFrame()

def fetch_live_prices():
    """Fetch current prices from CoinGecko"""
    try:
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
            return response.json()
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
            
            # Convert to dataframe
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
        "timestamp": datetime.now().isoformat()
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
        
        # Convert to records
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
            "count": len(transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exchange-flows")
def get_exchange_flows():
    """Get exchange inflow/outflow data"""
    try:
        df = fetch_dune_exchange_flows()
        
        if df.empty:
            return {"flows": [], "count": 0}
        
        # Group by exchange and token
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
        
        return {
            "flows": flows,
            "count": len(flows)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flow-summary")
def get_flow_summary():
    """Get aggregated flow summary"""
    try:
        df = fetch_dune_exchange_flows()
        
        if df.empty:
            return {
                "total_inflow": 0,
                "total_outflow": 0,
                "net_flow": 0,
                "sentiment": "neutral"
            }
        
        total_inflow = df['inflow'].sum()
        total_outflow = df['outflow'].sum()
        net_flow = total_outflow - total_inflow
        
        # Determine sentiment
        if net_flow > 0:
            sentiment = "bullish"  # More outflow = accumulation
        elif net_flow < 0:
            sentiment = "bearish"  # More inflow = selling pressure
        else:
            sentiment = "neutral"
        
        return {
            "total_inflow": float(total_inflow),
            "total_outflow": float(total_outflow),
            "net_flow": float(net_flow),
            "sentiment": sentiment,
            "by_token": df.groupby('token')[['inflow', 'outflow']].sum().to_dict('index')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/price-history")
def get_price_history_endpoint(days: int = 7):
    """Get ETH price history"""
    try:
        df = fetch_price_history(days)
        
        if df.empty:
            return {"prices": [], "count": 0}
        
        prices = []
        for _, row in df.iterrows():
            prices.append({
                "timestamp": row['timestamp'].isoformat(),
                "price": float(row['price'])
            })
        
        return {
            "prices": prices,
            "count": len(prices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/whale-price-overlay")
def get_whale_price_overlay():
    """Get whale transactions overlaid with price data"""
    try:
        # Fetch both datasets
        whales_df = fetch_dune_whale_data()
        prices_df = fetch_price_history(7)
        
        if whales_df.empty or prices_df.empty:
            return {"data": [], "count": 0}
        
        # Merge on timestamp (round to hour)
        whales_df['hour'] = whales_df['timestamp'].dt.floor('H')
        prices_df['hour'] = prices_df['timestamp'].dt.floor('H')
        
        merged = pd.merge(
            whales_df,
            prices_df[['hour', 'price']],
            on='hour',
            how='left'
        )
        
        data = []
        for _, row in merged.iterrows():
            data.append({
                "timestamp": row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None,
                "amount": float(row.get('amount', 0)),
                "token": row.get('token', ''),
                "price": float(row.get('price', 0)) if pd.notna(row.get('price')) else None,
                "tx_hash": row.get('tx_hash', '')
            })
        
        return {
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh-data")
def refresh_all_data():
    """Trigger a refresh of all data"""
    try:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
