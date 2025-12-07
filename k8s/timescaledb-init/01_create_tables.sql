-- enable TimescaleDB extension (needed before create_hypertable)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- then create tables
CREATE TABLE IF NOT EXISTS time_series_raw (
    series_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY(series_id, ts)
);

CREATE TABLE IF NOT EXISTS time_series_preprocessed (
    series_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION,
    features JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY(series_id, ts)
);

-- now create hypertables
SELECT create_hypertable('time_series_raw', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('time_series_preprocessed', 'ts', if_not_exists => TRUE);
