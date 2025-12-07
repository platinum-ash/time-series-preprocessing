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

# Generate dummy stock data for 'DUMMY'
start_date = datetime(2024, 1, 1)
num_days = 100

timestamps = [start_date + timedelta(days=i) for i in range(num_days)]
values = np.random.normal(loc=150, scale=5, size=num_days)  # dummy stock prices
metadata = [{} for _ in range(num_days)]  # empty metadata dicts

# Prepare rows for insertion
rows = [
    {
        "series_id": "DUMMY",
        "ts": ts,
        "value": float(val),
        "metadata": json.dumps(meta)
    }
    for ts, val, meta in zip(timestamps, values, metadata)
]

# Insert into database using SQLAlchemy
insert_stmt = text("""
    INSERT INTO time_series_raw (series_id, ts, value, metadata)
    VALUES (:series_id, :ts, :value, :metadata)
    ON CONFLICT (series_id, ts) DO UPDATE
        SET value = EXCLUDED.value,
            metadata = EXCLUDED.metadata;
""")

with engine.begin() as conn:
    conn.execute(insert_stmt, rows)

print(f"Inserted {len(rows)} dummy data points for AAPL")