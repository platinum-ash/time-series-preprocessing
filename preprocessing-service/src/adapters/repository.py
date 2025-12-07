"""
Repository adapters for data persistence.
These implement the ITimeSeriesRepository port.
"""

import pandas as pd
import json
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from typing import Optional
from src.domain.ports import ITimeSeriesRepository
from src.domain.models import TimeSeriesData


class TimescaleDBRepository(ITimeSeriesRepository):
    """
    TimescaleDB adapter for time series storage using SQLAlchemy.
    Supports OHLCV (Open, High, Low, Close, Volume) data format.
    """

    def __init__(self, connection_string: str):
        """
        Example connection string:
        postgresql+psycopg2://user:password@timescaledb:5432/timeseries
        """
        self.engine = create_engine(connection_string, pool_pre_ping=True)

        # Ensure hypertables exist (idempotent)
        self._initialize_schema()

    def _initialize_schema(self):
        """Create tables + hypertables if they do not exist."""
        with self.engine.begin() as conn:
            # Enable TimescaleDB extension
            conn.execute(text("""
            CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
            """))

            # Create raw data table
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS time_series_raw (
                series_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                features JSONB DEFAULT '{}'::jsonb,
                PRIMARY KEY(series_id, timestamp)
            );
            """))

            # Create preprocessed data table
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS time_series_preprocessed (
                series_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                features JSONB DEFAULT '{}'::jsonb,
                PRIMARY KEY(series_id, timestamp)
            );
            """))

            # Create hypertables
            conn.execute(text("""
            SELECT create_hypertable('time_series_raw', 'timestamp', if_not_exists => TRUE);
            """))

            conn.execute(text("""
            SELECT create_hypertable('time_series_preprocessed', 'timestamp', if_not_exists => TRUE);
            """))

    # -------------------------------
    # READ RAW DATA
    # -------------------------------
    def get_raw_data(self, series_id: str) -> TimeSeriesData:
        """
        Retrieve raw OHLCV data from TimescaleDB.
        """
        query = text("""
            SELECT timestamp, open, high, low, close, volume, features
            FROM time_series_raw
            WHERE series_id = :sid
            ORDER BY timestamp
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})

        if df.empty:
            raise ValueError(f"No raw data found for {series_id}")

        # Parse JSONB features if they're strings
        if 'features' in df.columns:
            df['features'] = df['features'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if x else {})
            )

        return TimeSeriesData.from_dataframe(
            df,
            metadata={"series_id": series_id}
        )

    # -------------------------------
    # SAVE RAW DATA
    # -------------------------------
    def save_raw_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """
        Save raw OHLCV data to TimescaleDB.
        """
        try:
            df = data.to_dataframe()

            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Ensure features column exists
            if 'features' not in df.columns:
                df['features'] = [{}] * len(df)

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

            with self.engine.begin() as conn:
                rows = [
                    {
                        "series_id": series_id,
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]) if pd.notna(row["open"]) else None,
                        "high": float(row["high"]) if pd.notna(row["high"]) else None,
                        "low": float(row["low"]) if pd.notna(row["low"]) else None,
                        "close": float(row["close"]) if pd.notna(row["close"]) else None,
                        "volume": float(row["volume"]) if pd.notna(row["volume"]) else None,
                        "features": json.dumps(row["features"]) if isinstance(row["features"], dict) else json.dumps({})
                    }
                    for _, row in df.iterrows()
                ]
                
                conn.execute(insert_stmt, rows)

            return True

        except SQLAlchemyError as e:
            print(f"Error saving raw data: {e}")
            import traceback
            traceback.print_exc()
            return False

    # -------------------------------
    # SAVE PREPROCESSED DATA
    # -------------------------------
    def save_preprocessed_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """
        Save preprocessed OHLCV data to TimescaleDB.
        Features are stored as JSONB - one JSON object per timestamp.
        """
        try:
            df = data.to_dataframe()

            # Debug logging
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            print(f"DEBUG: 'features' in columns: {'features' in df.columns}")
            if 'features' in df.columns and len(df) > 0:
                print(f"DEBUG: First features value: {df['features'].iloc[0]}")
                print(f"DEBUG: Features type: {type(df['features'].iloc[0])}")

            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Ensure features column exists
            if 'features' not in df.columns:
                df['features'] = [{}] * len(df)

            insert_stmt = text("""
            INSERT INTO time_series_preprocessed 
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

            with self.engine.begin() as conn:
                rows = [
                    {
                        "series_id": series_id,
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]) if pd.notna(row["open"]) else None,
                        "high": float(row["high"]) if pd.notna(row["high"]) else None,
                        "low": float(row["low"]) if pd.notna(row["low"]) else None,
                        "close": float(row["close"]) if pd.notna(row["close"]) else None,
                        "volume": float(row["volume"]) if pd.notna(row["volume"]) else None,
                        "features": json.dumps(row["features"]) if isinstance(row["features"], dict) else json.dumps({})
                    }
                    for _, row in df.iterrows()
                ]
                
                # Debug: print first row
                if rows:
                    print(f"DEBUG: First row to insert: {rows[0]}")
                
                conn.execute(insert_stmt, rows)

            return True

        except SQLAlchemyError as e:
            print(f"Error saving preprocessed data: {e}")
            import traceback
            traceback.print_exc()
            return False

    # -------------------------------
    # READ PREPROCESSED DATA
    # -------------------------------
    def get_preprocessed_data(self, series_id: str) -> TimeSeriesData:
        """
        Retrieve preprocessed OHLCV data with features from TimescaleDB.
        Features are parsed from JSONB back into Python dicts.
        """
        query = text("""
            SELECT timestamp, open, high, low, close, volume, features
            FROM time_series_preprocessed
            WHERE series_id = :sid
            ORDER BY timestamp
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})

        if df.empty:
            raise ValueError(f"No preprocessed data found for {series_id}")
        
        # Parse JSONB features back to Python dicts (if they're strings)
        if 'features' in df.columns:
            df['features'] = df['features'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if x else {})
            )

        return TimeSeriesData.from_dataframe(df, {"series_id": series_id})
    
    # -------------------------------
    # QUERY FEATURES
    # -------------------------------
    def get_feature_names(self, series_id: str, table: str = 'preprocessed') -> list:
        """
        Get all unique feature names for a series.
        Useful for understanding what features are available.
        
        Args:
            series_id: Time series identifier
            table: Either 'raw' or 'preprocessed'
        """
        table_name = f"time_series_{table}"
        query = text(f"""
            SELECT DISTINCT jsonb_object_keys(features) as feature_name
            FROM {table_name}
            WHERE series_id = :sid
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id})
            return [row[0] for row in result]
    
    def get_data_with_specific_features(
        self, 
        series_id: str, 
        feature_names: list,
        table: str = 'preprocessed'
    ) -> pd.DataFrame:
        """
        Get OHLCV data with only specific features extracted.
        
        Args:
            series_id: Time series identifier
            feature_names: List of feature names to extract from JSONB
            table: Either 'raw' or 'preprocessed'
            
        Returns:
            DataFrame with timestamp, OHLCV columns, and selected features as columns
        """
        table_name = f"time_series_{table}"
        
        # Build feature selection for each requested feature
        feature_selects = [
            f"features->'{name}' as {name}" 
            for name in feature_names
        ]
        feature_sql = ", ".join(feature_selects) if feature_selects else ""
        
        # Build column list
        base_cols = "timestamp, open, high, low, close, volume"
        select_cols = f"{base_cols}, {feature_sql}" if feature_sql else base_cols
        
        query = text(f"""
            SELECT {select_cols}
            FROM {table_name}
            WHERE series_id = :sid
            ORDER BY timestamp
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})
        
        return df
    
    # -------------------------------
    # UTILITY METHODS
    # -------------------------------
    def get_date_range(self, series_id: str, table: str = 'raw') -> tuple:
        """
        Get the earliest and latest timestamps for a series.
        
        Args:
            series_id: Time series identifier
            table: Either 'raw' or 'preprocessed'
            
        Returns:
            Tuple of (earliest_timestamp, latest_timestamp)
        """
        table_name = f"time_series_{table}"
        query = text(f"""
            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM {table_name}
            WHERE series_id = :sid
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id}).fetchone()
            return (result[0], result[1]) if result else (None, None)
    
    def get_series_count(self, series_id: str, table: str = 'raw') -> int:
        """
        Get the number of records for a series.
        
        Args:
            series_id: Time series identifier
            table: Either 'raw' or 'preprocessed'
            
        Returns:
            Number of records
        """
        table_name = f"time_series_{table}"
        query = text(f"""
            SELECT COUNT(*) as count
            FROM {table_name}
            WHERE series_id = :sid
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id}).fetchone()
            return result[0] if result else 0