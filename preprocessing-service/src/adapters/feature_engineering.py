"""
Adapters for feature engineering operations.
"""
import pandas as pd
import numpy as np
from typing import List
from src.domain.ports import IFeatureEngineer
from src.domain.models import TimeSeriesData


class FeatureEngineer(IFeatureEngineer):
    """
    Feature engineer using pandas for creating lag and rolling features.
    Now supports OHLCV data structure.
    """
    
    def __init__(self, price_column: str = 'close'):
        """
        Args:
            price_column: Which price column to use for lag/rolling features
                         Options: 'open', 'high', 'low', 'close'
        """
        self.price_column = price_column
    
    def create_lag_features(
        self, 
        data: TimeSeriesData, 
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged features from the specified price column.
        
        Args:
            data: Time series data with OHLCV columns
            lags: List of lag periods (e.g., [1, 7, 30])
            
        Returns:
            DataFrame with lag features
        """
        df = data.to_dataframe()
        lag_features = pd.DataFrame(index=df.index)
        
        # Create lag features for the configured price column
        if self.price_column not in df.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        price_values = df[self.price_column]
        
        for lag in lags:
            lag_features[f'lag_{lag}'] = price_values.shift(lag)
        
        return lag_features
    
    def create_rolling_features(
        self, 
        data: TimeSeriesData, 
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Create rolling window statistics features from the specified price column.
        
        Args:
            data: Time series data with OHLCV columns
            windows: List of window sizes (e.g., [7, 30, 90])
            
        Returns:
            DataFrame with rolling features (mean, std, min, max)
        """
        df = data.to_dataframe()
        rolling_features = pd.DataFrame(index=df.index)
        
        # Create rolling features for the configured price column
        if self.price_column not in df.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        price_values = df[self.price_column]
        
        for window in windows:
            rolling = price_values.rolling(window=window)
            rolling_features[f'rolling_mean_{window}'] = rolling.mean()
            rolling_features[f'rolling_std_{window}'] = rolling.std()
            rolling_features[f'rolling_min_{window}'] = rolling.min()
            rolling_features[f'rolling_max_{window}'] = rolling.max()
        
        return rolling_features
    
    def create_time_features(self, data: TimeSeriesData) -> pd.DataFrame:
        """
        Create time-based features (hour, day, month, etc.)
        
        Args:
            data: Time series data
            
        Returns:
            DataFrame with time-based features
        """
        df = data.to_dataframe()
        time_features = pd.DataFrame(index=df.index)
        
        timestamps = pd.to_datetime(df['timestamp'])
        
        time_features['hour'] = timestamps.dt.hour
        time_features['day_of_week'] = timestamps.dt.dayofweek
        time_features['day_of_month'] = timestamps.dt.day
        time_features['month'] = timestamps.dt.month
        time_features['quarter'] = timestamps.dt.quarter
        time_features['year'] = timestamps.dt.year
        time_features['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
        
        # Cyclical encoding for periodic features
        time_features['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
        time_features['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)
        time_features['day_of_week_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        time_features['day_of_week_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        
        return time_features
    
    def create_ohlcv_features(self, data: TimeSeriesData) -> pd.DataFrame:
        """
        Create OHLCV-specific technical features.
        
        Args:
            data: Time series data with OHLCV columns
            
        Returns:
            DataFrame with OHLCV-derived technical indicators
        """
        df = data.to_dataframe()
        features = pd.DataFrame(index=df.index)
        
        # Price range features
        features['price_range'] = df['high'] - df['low']
        features['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Body and wick features (candlestick analysis)
        features['body'] = df['close'] - df['open']
        features['body_pct'] = (df['close'] - df['open']) / df['open'] * 100
        features['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        features['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Price position within range (avoid division by zero)
        price_range = df['high'] - df['low']
        features['close_position'] = np.where(
            price_range > 0,
            (df['close'] - df['low']) / price_range,
            0.5  # Default to middle if no range
        )
        
        # Volume-weighted average price (VWAP)
        features['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
        
        # Typical price (used in many technical indicators)
        features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Price momentum
        features['close_change'] = df['close'].diff()
        features['close_pct_change'] = df['close'].pct_change() * 100
        
        # Volume features
        features['volume_change'] = df['volume'].diff()
        features['volume_pct_change'] = df['volume'].pct_change() * 100
        
        # Price-volume relationship
        features['volume_price_trend'] = (df['close'].pct_change() * df['volume']).fillna(0)
        
        # True Range (used in ATR calculation)
        prev_close = df['close'].shift(1)
        features['true_range'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        # Gap detection
        features['gap'] = df['open'] - prev_close
        features['gap_pct'] = (df['open'] - prev_close) / prev_close * 100
        
        return features