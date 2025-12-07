"""
Adapters for outlier detection in time series.
"""
import pandas as pd
import numpy as np
from typing import List
from src.domain.ports import IOutlierDetector
from src.domain.models import TimeSeriesData, OutlierMethod


class StatisticalOutlierDetector(IOutlierDetector):
    """
    Outlier detector using statistical methods (Z-score, IQR, Isolation Forest).
    Now supports OHLCV data structure.
    """
    
    def detect_and_remove(
        self, 
        data: TimeSeriesData, 
        method: OutlierMethod, 
        threshold: float,
        price_column: str = 'close'
    ) -> TimeSeriesData:
        """
        Detect and remove outliers from OHLCV time series.
        
        Args:
            data: Time series data with OHLCV columns
            method: Method for outlier detection
            threshold: Threshold for outlier detection (used by ZSCORE and IQR)
            price_column: Which price column to use for outlier detection
            
        Returns:
            Time series data with outliers removed
        """
        df = data.to_dataframe()
        
        # Validate price column exists
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        price_values = df[price_column]
        
        # Detect outliers based on method
        if method == OutlierMethod.ZSCORE:
            z_scores = np.abs(
                (price_values - price_values.mean()) / price_values.std()
            )
            mask = z_scores < threshold
            df = df[mask]
        
        elif method == OutlierMethod.IQR:
            Q1 = price_values.quantile(0.25)
            Q3 = price_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[
                (price_values >= lower_bound) & 
                (price_values <= upper_bound)
            ]
        
        elif method == OutlierMethod.ISOLATION_FOREST:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            values = price_values.values.reshape(-1, 1)
            
            # Train isolation forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            predictions = iso_forest.fit_predict(values)
            
            # Keep only inliers (prediction == 1)
            df = df[predictions == 1]
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Preserve features if they exist in the filtered data
        return TimeSeriesData.from_dataframe(df, data.metadata)
    
    def detect_only(
        self, 
        data: TimeSeriesData, 
        method: OutlierMethod, 
        threshold: float,
        price_column: str = 'close'
    ) -> List[int]:
        """
        Detect outlier indices without removing them.
        
        Args:
            data: Time series data with OHLCV columns
            method: Method for outlier detection
            threshold: Threshold for outlier detection
            price_column: Which price column to use for outlier detection
            
        Returns:
            List of indices where outliers were detected
        """
        df = data.to_dataframe()
        
        # Validate price column exists
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        price_values = df[price_column]
        outlier_mask = np.zeros(len(df), dtype=bool)
        
        if method == OutlierMethod.ZSCORE:
            z_scores = np.abs(
                (price_values - price_values.mean()) / price_values.std()
            )
            outlier_mask = z_scores >= threshold
        
        elif method == OutlierMethod.IQR:
            Q1 = price_values.quantile(0.25)
            Q3 = price_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (price_values < lower_bound) | (price_values > upper_bound)
        
        elif method == OutlierMethod.ISOLATION_FOREST:
            from sklearn.ensemble import IsolationForest
            
            values = price_values.values.reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(values)
            outlier_mask = predictions == -1
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return np.where(outlier_mask)[0].tolist()