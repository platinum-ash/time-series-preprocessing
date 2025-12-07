# /app/src/populate_dummy_data.py
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Load database URL from environment
import os
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://tsuser:ts_password@localhost:5432/timeseries")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

def generate_ohlcv_data(num_days: int, initial_price: float = 150.0, volatility: float = 0.02):
    """
    Generate realistic OHLCV (Open, High, Low, Close, Volume) data.
    
    Args:
        num_days: Number of days to generate
        initial_price: Starting price
        volatility: Daily volatility (standard deviation as percentage)
    
    Returns:
        Dictionary with lists for open, high, low, close, volume
    """
    np.random.seed(42)  # For reproducibility
    
    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }
    
    current_price = initial_price
    
    for _ in range(num_days):
        # Generate daily return
        daily_return = np.random.normal(0, volatility)
        
        # Open price (close from previous day with small gap)
        open_price = current_price * (1 + np.random.normal(0, volatility * 0.3))
        
        # Generate intraday movement
        high_price = open_price * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low_price = open_price * (1 - abs(np.random.normal(0, volatility * 0.5)))
        
        # Close price
        close_price = open_price * (1 + daily_return)
        
        # Ensure high is the highest and low is the lowest
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (random around a mean with some variation)
        base_volume = 1_000_000
        volume = int(np.random.lognormal(np.log(base_volume), 0.5))
        
        data['open'].append(round(open_price, 2))
        data['high'].append(round(high_price, 2))
        data['low'].append(round(low_price, 2))
        data['close'].append(round(close_price, 2))
        data['volume'].append(volume)
        
        # Update current price for next day
        current_price = close_price
    
    return data

# Generate dummy stock data for multiple series
series_configs = [
    {"series_id": "AAPL", "initial_price": 150.0, "volatility": 0.02},
    {"series_id": "GOOGL", "initial_price": 2800.0, "volatility": 0.025},
    {"series_id": "MSFT", "initial_price": 350.0, "volatility": 0.018},
    {"series_id": "DUMMY", "initial_price": 100.0, "volatility": 0.03},
]

start_date = datetime(2024, 1, 1)
num_days = 100

# Generate timestamps
timestamps = [start_date + timedelta(days=i) for i in range(num_days)]

# Prepare all rows for insertion
all_rows = []

for config in series_configs:
    print(f"Generating data for {config['series_id']}...")
    
    # Generate OHLCV data
    ohlcv_data = generate_ohlcv_data(
        num_days=num_days,
        initial_price=config['initial_price'],
        volatility=config['volatility']
    )
    
    # Create rows for this series
    for i, ts in enumerate(timestamps):
        row = {
            "series_id": config['series_id'],
            "timestamp": ts,
            "open": float(ohlcv_data['open'][i]),
            "high": float(ohlcv_data['high'][i]),
            "low": float(ohlcv_data['low'][i]),
            "close": float(ohlcv_data['close'][i]),
            "volume": float(ohlcv_data['volume'][i]),
            "features": json.dumps({})  # Empty features for raw data
        }
        all_rows.append(row)

# Insert into database using SQLAlchemy
insert_stmt = text("""
    INSERT INTO time_series_raw 
        (series_id, timestamp, open, high, low, close, volume, features)
    VALUES 
        (:series_id, :timestamp, :open, :high, :low, :close, :volume, :features)
    ON CONFLICT (series_id, timestamp) DO UPDATE
    SET open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        features = EXCLUDED.features;
""")

print(f"\nInserting {len(all_rows)} total data points into database...")

with engine.begin() as conn:
    conn.execute(insert_stmt, all_rows)

print(f"✓ Successfully inserted {len(all_rows)} data points")
print(f"✓ Series populated: {', '.join([c['series_id'] for c in series_configs])}")
print(f"✓ Date range: {timestamps[0].date()} to {timestamps[-1].date()}")

# Optional: Print summary statistics for each series
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for config in series_configs:
    query = text("""
        SELECT 
            series_id,
            COUNT(*) as num_records,
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            ROUND(AVG(close)::numeric, 2) as avg_close,
            ROUND(MIN(low)::numeric, 2) as min_price,
            ROUND(MAX(high)::numeric, 2) as max_price,
            ROUND(AVG(volume)::numeric, 0) as avg_volume
        FROM time_series_raw
        WHERE series_id = :sid
        GROUP BY series_id
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"sid": config['series_id']}).fetchone()
        if result:
            print(f"\n{result[0]}:")
            print(f"  Records: {result[1]}")
            print(f"  Date Range: {result[2]} to {result[3]}")
            print(f"  Avg Close: ${result[4]}")
            print(f"  Price Range: ${result[5]} - ${result[6]}")
            print(f"  Avg Volume: {int(result[7]):,}")

print("\n" + "="*60)