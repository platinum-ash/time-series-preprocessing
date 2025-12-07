"""
Adapter for handling missing values in time series.
"""
import pandas as pd
from src.domain.models import TimeSeriesData, InterpolationMethod
from src.domain.ports import IMissingValueHandler


class MissingValueHandler(IMissingValueHandler):
    """
    Implementation for handling missing values using pandas interpolation.
    Now supports OHLCV data structure.
    """
    
    def handle_missing(
        self, 
        data: TimeSeriesData, 
        method: InterpolationMethod
    ) -> TimeSeriesData:
        """
        Handle missing values in OHLCV time series data.
        Applies interpolation to all price and volume columns.
        
        Args:
            data: Time series data with potential missing values
            method: Interpolation method to use
            
        Returns:
            Time series data with missing values handled
        """
        df = data.to_dataframe()
        
        # Define OHLCV columns to interpolate
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Apply interpolation based on method
        if method == InterpolationMethod.LINEAR:
            for col in ohlcv_columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear')
                    
        elif method == InterpolationMethod.FORWARD_FILL:
            for col in ohlcv_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')
                    
        elif method == InterpolationMethod.BACKWARD_FILL:
            for col in ohlcv_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(method='bfill')
                    
        elif method == InterpolationMethod.SPLINE:
            for col in ohlcv_columns:
                if col in df.columns:
                    # Spline requires at least order+1 non-NaN values
                    if df[col].notna().sum() > 3:
                        df[col] = df[col].interpolate(method='spline', order=3)
                    else:
                        # Fallback to linear if not enough data
                        df[col] = df[col].interpolate(method='linear')
                        
        elif method == InterpolationMethod.POLYNOMIAL:
            for col in ohlcv_columns:
                if col in df.columns:
                    # Polynomial requires at least order+1 non-NaN values
                    if df[col].notna().sum() > 2:
                        df[col] = df[col].interpolate(method='polynomial', order=2)
                    else:
                        # Fallback to linear if not enough data
                        df[col] = df[col].interpolate(method='linear')
        
        # Handle any remaining NaN values at the edges with forward/backward fill
        for col in ohlcv_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Preserve features if they exist
        if 'features' not in df.columns and data.features is not None:
            df['features'] = data.features
        
        return TimeSeriesData.from_dataframe(df, data.metadata)