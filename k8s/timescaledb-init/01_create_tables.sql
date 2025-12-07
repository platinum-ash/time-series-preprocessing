-- enable TimescaleDB extension (needed before create_hypertable)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
 
-- then create tables
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
 
-- now create hypertables
SELECT create_hypertable('time_series_raw', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('time_series_preprocessed', 'timestamp', if_not_exists => TRUE);