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

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS time_series_raw (
                series_id TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                value DOUBLE PRECISION,
                metadata JSONB DEFAULT '{}'::jsonb,
                PRIMARY KEY(series_id, ts)
            );
            """))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS time_series_preprocessed (
                series_id TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                value DOUBLE PRECISION,
                features JSONB DEFAULT '{}'::jsonb,
                PRIMARY KEY(series_id, ts)
            );
            """))

            conn.execute(text("""
            SELECT create_hypertable('time_series_raw', 'ts', if_not_exists => TRUE);
            """))

            conn.execute(text("""
            SELECT create_hypertable('time_series_preprocessed', 'ts', if_not_exists => TRUE);
            """))

    # -------------------------------
    # READ RAW DATA
    # -------------------------------
    def get_raw_data(self, series_id: str) -> TimeSeriesData:
        query = text("""
            SELECT ts, value, metadata
            FROM time_series_raw
            WHERE series_id = :sid
            ORDER BY ts
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})

        if df.empty:
            raise ValueError(f"No raw data found for {series_id}")

        # Normalize DataFrame to match your domain model
        df.rename(columns={"ts": "timestamp"}, inplace=True)

        return TimeSeriesData.from_dataframe(
            df,
            metadata={"series_id": series_id}
        )

    # -------------------------------
    # SAVE PREPROCESSED DATA
    # -------------------------------
    def save_preprocessed_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """
        Save preprocessed data to TimescaleDB.
        Features are stored as JSONB - one JSON object per timestamp.
        """
        try:
            df = data.to_dataframe()

            # Debug logging
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            print(f"DEBUG: 'features' in columns: {'features' in df.columns}")
            if 'features' in df.columns:
                print(f"DEBUG: First features value: {df['features'].iloc[0]}")
                print(f"DEBUG: Features type: {type(df['features'].iloc[0])}")

            # Ensure there is a 'features' column (empty JSON if missing)
            if 'features' not in df.columns:
                df['features'] = [{}] * len(df)

            insert_stmt = text("""
            INSERT INTO time_series_preprocessed (series_id, ts, value, features)
            VALUES (:series_id, :ts, :value, :features)
            ON CONFLICT (series_id, ts) DO UPDATE
            SET value = EXCLUDED.value,
                features = EXCLUDED.features;
            """)

            with self.engine.begin() as conn:
                # Prepare list of dicts for bulk execution
                rows = [
                    {
                        "series_id": series_id,
                        "ts": row["timestamp"],
                        "value": float(row["value"]),
                        # FIX: Use bracket notation, not .get()
                        "features": json.dumps(row["features"]) if isinstance(row["features"], dict) else json.dumps({})
                    }
                    for _, row in df.iterrows()
                ]
                
                # Debug: print first row
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
        Retrieve preprocessed data with features from TimescaleDB.
        Features are parsed from JSONB back into Python dicts.
        """
        query = text("""
            SELECT ts, value, features
            FROM time_series_preprocessed
            WHERE series_id = :sid
            ORDER BY ts
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})

        if df.empty:
            raise ValueError(f"No preprocessed data found for {series_id}")

        df.rename(columns={"ts": "timestamp"}, inplace=True)
        
        # Parse JSONB features back to Python dicts (if they're strings)
        if 'features' in df.columns:
            df['features'] = df['features'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        return TimeSeriesData.from_dataframe(df, {"series_id": series_id})
    
    # -------------------------------
    # QUERY FEATURES
    # -------------------------------
    def get_feature_names(self, series_id: str) -> list:
        """
        Get all unique feature names for a series.
        Useful for understanding what features are available.
        """
        query = text("""
            SELECT DISTINCT jsonb_object_keys(features) as feature_name
            FROM time_series_preprocessed
            WHERE series_id = :sid
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id})
            return [row[0] for row in result]
    
    def get_data_with_specific_features(
        self, 
        series_id: str, 
        feature_names: list
    ) -> pd.DataFrame:
        """
        Get preprocessed data with only specific features extracted.
        
        Args:
            series_id: Time series identifier
            feature_names: List of feature names to extract from JSONB
            
        Returns:
            DataFrame with timestamp, value, and selected features as columns
        """
        # Build feature selection for each requested feature
        feature_selects = [
            f"features->'{name}' as {name}" 
            for name in feature_names
        ]
        feature_sql = ", ".join(feature_selects)
        
        query = text(f"""
            SELECT ts, value, {feature_sql}
            FROM time_series_preprocessed
            WHERE series_id = :sid
            ORDER BY ts
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})
        
        df.rename(columns={"ts": "timestamp"}, inplace=True)
        return df