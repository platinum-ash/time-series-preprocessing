"""
Adapter for resampling time series data.
"""
import pandas as pd

from src.domain.models import TimeSeriesData, AggregationMethod
from src.domain.ports import IResampler


class Resampler(IResampler):
    """
    Implementation for resampling time series data to different frequencies.
    Now supports OHLCV data with proper OHLC aggregation rules.
    """
    
    def resample(
        self,
        data: TimeSeriesData,
        frequency: str,
        aggregation: AggregationMethod
    ) -> TimeSeriesData:
        """
        Resample OHLCV time series data to a different frequency.
        
        For OHLCV data, uses proper financial data aggregation:
        - Open: first value in period
        - High: maximum value in period
        - Low: minimum value in period
        - Close: last value in period
        - Volume: sum of volume in period
        
        Args:
            data: Time series data
            frequency: Pandas frequency string (e.g., '1D', '1W', '1M')
            aggregation: Aggregation method (used for features, not OHLCV)
            
        Returns:
            Resampled time series data
        """
        df = data.to_dataframe()
        df.set_index('timestamp', inplace=True)
        
        # Define proper OHLCV aggregation rules
        ohlcv_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Apply OHLCV aggregation
        resampled = df[['open', 'high', 'low', 'close', 'volume']].resample(frequency).agg(ohlcv_agg)
        
        # Drop rows with NaN (periods with no data)
        resampled = resampled.dropna()
        
        # Reset index to get timestamp back as a column
        resampled.reset_index(inplace=True)
        
        # Note: features are typically dropped during resampling as they may not be meaningful
        # If you need to preserve features, you'd need to define custom aggregation logic
        
        return TimeSeriesData.from_dataframe(resampled, data.metadata)
    
    def _get_aggregation_func(self, method: AggregationMethod):
        """
        Get the pandas aggregation function for a given method.
        
        Args:
            method: Aggregation method enum
            
        Returns:
            String name of pandas aggregation function
        """
        mapping = {
            AggregationMethod.MEAN: 'mean',
            AggregationMethod.SUM: 'sum',
            AggregationMethod.MIN: 'min',
            AggregationMethod.MAX: 'max',
            AggregationMethod.MEDIAN: 'median'
        }
        return mapping.get(method, 'mean')