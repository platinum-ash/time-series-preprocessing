"""
Domain models for the preprocessing service.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import pandas as pd


class InterpolationMethod(Enum):
    """Methods for handling missing values"""
    LINEAR = "linear"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"


class OutlierMethod(Enum):
    """Methods for detecting outliers"""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class AggregationMethod(Enum):
    """Methods for aggregating resampled data"""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


@dataclass
class TimeSeriesData:
    """Domain model for OHLCV time series data"""
    timestamps: List[datetime]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    metadata: dict = field(default_factory=dict)
    features: Optional[List[dict]] = None
    
    # Backward compatibility: provide 'values' as alias for 'close'
    @property
    def values(self) -> List[float]:
        """Alias for close prices for backward compatibility"""
        return self.close
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, metadata: dict = None) -> 'TimeSeriesData':
        """Create TimeSeriesData from DataFrame with OHLCV columns"""
        # Extract features if present
        features = None
        if 'features' in df.columns:
            features = df['features'].tolist()
        
        # Check if we have OHLCV data or legacy single-value data
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # OHLCV format
            return cls(
                timestamps=df['timestamp'].tolist(),
                open=df['open'].tolist(),
                high=df['high'].tolist(),
                low=df['low'].tolist(),
                close=df['close'].tolist(),
                volume=df['volume'].tolist(),
                metadata=metadata or {},
                features=features
            )
        elif 'value' in df.columns:
            # Legacy single-value format - use value for all OHLC, set volume to 0
            values = df['value'].tolist()
            return cls(
                timestamps=df['timestamp'].tolist(),
                open=values,
                high=values,
                low=values,
                close=values,
                volume=[0.0] * len(values),
                metadata=metadata or {},
                features=features
            )
        else:
            raise ValueError("DataFrame must contain either OHLCV columns or 'value' column")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with OHLCV columns"""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })
        
        # Add features if they exist
        if self.features is not None:
            df['features'] = self.features
        
        return df
    
    def get_price_column(self, column: str = 'close') -> List[float]:
        """
        Get a specific price column for processing.
        
        Args:
            column: One of 'open', 'high', 'low', 'close', or 'volume'
            
        Returns:
            List of values for the specified column
        """
        column_map = {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
        
        if column not in column_map:
            raise ValueError(f"Invalid column: {column}. Must be one of {list(column_map.keys())}")
        
        return column_map[column]
    
    def __len__(self):
        return len(self.close)


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing operations.
    Encapsulates all parameters needed for the preprocessing pipeline.
    """
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR
    outlier_method: OutlierMethod = OutlierMethod.ZSCORE
    outlier_threshold: float = 3.0
    resample_frequency: str = None
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    lag_features: List[int] = None
    rolling_window_sizes: List[int] = None
    price_column: str = 'close'  # Which price column to use for features/outlier detection
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.outlier_threshold <= 0:
            raise ValueError("Outlier threshold must be positive")
        
        if self.lag_features and any(lag < 1 for lag in self.lag_features):
            raise ValueError("Lag values must be positive integers")
        
        if self.rolling_window_sizes and any(w < 2 for w in self.rolling_window_sizes):
            raise ValueError("Rolling window sizes must be at least 2")
        
        valid_price_columns = ['open', 'high', 'low', 'close']
        if self.price_column not in valid_price_columns:
            raise ValueError(f"price_column must be one of {valid_price_columns}")