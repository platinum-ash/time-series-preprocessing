"""
Core business logic for preprocessing time series data.
"""
from typing import Optional
import pandas as pd
import json
from .models import TimeSeriesData, PreprocessingConfig
from .ports import (
    ITimeSeriesRepository,
    IMissingValueHandler,
    IOutlierDetector,
    IFeatureEngineer,
    IResampler,
    ILogger
)


class PreprocessingService:
    """
    Core preprocessing service implementing the business logic.
    Uses dependency injection - all dependencies are ports (interfaces).
    """
    
    def __init__(
        self,
        repository: ITimeSeriesRepository,
        missing_handler: IMissingValueHandler,
        outlier_detector: IOutlierDetector,
        feature_engineer: IFeatureEngineer,
        resampler: IResampler,
        logger: ILogger
    ):
        self.repository = repository
        self.missing_handler = missing_handler
        self.outlier_detector = outlier_detector
        self.feature_engineer = feature_engineer
        self.resampler = resampler
        self.logger = logger
    
    def preprocess(
        self, 
        series_id: str, 
        config: PreprocessingConfig
    ) -> TimeSeriesData:
        """
        Execute the complete preprocessing pipeline with feature engineering.
        
        Args:
            series_id: Unique identifier for the time series
            config: Preprocessing configuration
            
        Returns:
            Preprocessed time series data with features
        """
        try:
            self.logger.info(f"Starting preprocessing for series: {series_id}")
            
            # Step 1: Retrieve raw data
            data = self.repository.get_raw_data(series_id)
            original_count = len(data)
            self.logger.info(f"Retrieved {original_count} data points")
            
            # Step 2: Handle missing values (apply to all OHLCV columns)
            data = self.missing_handler.handle_missing(
                data, 
                config.interpolation_method
            )
            self.logger.info(
                f"Missing values handled using {config.interpolation_method.value}"
            )
            
            # Step 3: Detect and remove outliers (based on configured price column)
            data = self.outlier_detector.detect_and_remove(
                data, 
                config.outlier_method, 
                config.outlier_threshold,
                config.price_column
            )
            removed_count = original_count - len(data)
            self.logger.info(
                f"Outliers processed: {removed_count} points removed "
                f"using {config.outlier_method.value}"
            )
            
            # Step 4: Resample if frequency specified
            if config.resample_frequency:
                data = self.resampler.resample(
                    data, 
                    config.resample_frequency, 
                    config.aggregation_method
                )
                self.logger.info(
                    f"Resampled to {config.resample_frequency} "
                    f"using {config.aggregation_method.value}"
                )
            
            # Step 5: Feature Engineering
            features_dict = self._create_features_dict(data, config)
            
            # Attach features to the data
            data = self._attach_features_to_data(data, features_dict)
            self.logger.info(f"Created {len(features_dict)} feature columns")
            
            # Step 6: Save preprocessed data with features
            success = self.repository.save_preprocessed_data(series_id, data)
            if success:
                self.logger.info("Preprocessing completed successfully")
            else:
                self.logger.warning("Failed to save preprocessed data")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {series_id}", e)
            raise
    
    def _create_features_dict(
        self, 
        data: TimeSeriesData, 
        config: PreprocessingConfig
    ) -> dict:
        """
        Create all features and organize them into a dictionary structure.
        
        Returns:
            Dictionary mapping feature names to their DataFrames
        """
        features = {}
        
        # Debug: Log what config values we have
        self.logger.info(f"Feature config - lag_features: {config.lag_features}")
        self.logger.info(f"Feature config - rolling_window_sizes: {config.rolling_window_sizes}")
        self.logger.info(f"Feature config - price_column: {config.price_column}")
        
        # Add lag features
        if config.lag_features:
            lag_df = self.feature_engineer.create_lag_features(
                data, 
                config.lag_features
            )
            self.logger.info(f"Lag features shape: {lag_df.shape}, columns: {lag_df.columns.tolist()}")
            for col in lag_df.columns:
                features[col] = lag_df[col]
            self.logger.info(f"Created {len(config.lag_features)} lag features")
        
        # Add rolling features
        if config.rolling_window_sizes:
            rolling_df = self.feature_engineer.create_rolling_features(
                data, 
                config.rolling_window_sizes
            )
            self.logger.info(f"Rolling features shape: {rolling_df.shape}, columns: {rolling_df.columns.tolist()}")
            for col in rolling_df.columns:
                features[col] = rolling_df[col]
            self.logger.info(
                f"Created rolling features for {len(config.rolling_window_sizes)} windows"
            )
        
        # Add time-based features
        time_df = self.feature_engineer.create_time_features(data)
        self.logger.info(f"Time features shape: {time_df.shape}, columns: {time_df.columns.tolist()}")
        for col in time_df.columns:
            features[col] = time_df[col]
        self.logger.info("Created time-based features")
        
        # Add OHLCV-specific features
        ohlcv_features = self._create_ohlcv_features(data)
        for col in ohlcv_features.columns:
            features[col] = ohlcv_features[col]
        self.logger.info(f"Created {len(ohlcv_features.columns)} OHLCV-specific features")
        
        # Debug: Log total features created
        self.logger.info(f"Total features dictionary keys: {list(features.keys())}")
        
        return features
    
    def _create_ohlcv_features(self, data: TimeSeriesData) -> pd.DataFrame:
        """
        Create OHLCV-specific technical features.
        
        Returns:
            DataFrame with OHLCV-derived features
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
        
        # Price position within range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume-weighted features
        features['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
        
        # Typical price (used in many technical indicators)
        features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return features
    
    def _attach_features_to_data(
        self, 
        data: TimeSeriesData, 
        features_dict: dict
    ) -> TimeSeriesData:
        """
        Attach engineered features to the TimeSeriesData object.
        
        Features are stored as JSONB in the database, one JSON object per timestamp.
        """
        df = data.to_dataframe()
        
        # Debug log
        self.logger.info(f"Original dataframe shape: {df.shape}")
        self.logger.info(f"Features dict has {len(features_dict)} keys")
        
        # Create a features column with JSON objects for each row
        if features_dict:
            # Convert the features dict (which has Series as values) into a DataFrame
            features_df = pd.DataFrame(features_dict)
            
            self.logger.info(f"Features dataframe shape: {features_df.shape}")
            self.logger.info(f"Features dataframe columns: {features_df.columns.tolist()}")
            
            # Convert each row to a dictionary (JSON object)
            # Handle NaN values by converting to None
            df['features'] = features_df.apply(
                lambda row: {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float)) else v) 
                            for k, v in row.to_dict().items()}, 
                axis=1
            )
            
            self.logger.info(f"Sample feature object: {df['features'].iloc[0]}")
        else:
            # Empty features if none were created
            self.logger.warning("No features were created, using empty dict")
            df['features'] = [{}] * len(df)
        
        # Update the data object with the features column
        data = TimeSeriesData.from_dataframe(df, data.metadata)
        
        return data
    
    def create_features(
        self, 
        series_id: str, 
        config: PreprocessingConfig
    ) -> pd.DataFrame:
        """
        Create engineered features from time series data.
        
        Args:
            series_id: Unique identifier for the time series
            config: Configuration with feature specifications
            
        Returns:
            DataFrame with original data and engineered features
        """
        try:
            self.logger.info(f"Creating features for series: {series_id}")
            
            # Get preprocessed data if available, otherwise raw data
            try:
                data = self.repository.get_preprocessed_data(series_id)
            except:
                data = self.repository.get_raw_data(series_id)
            
            df = data.to_dataframe()
            
            # Add lag features
            if config.lag_features:
                lag_df = self.feature_engineer.create_lag_features(
                    data, 
                    config.lag_features
                )
                df = pd.concat([df, lag_df], axis=1)
                self.logger.info(f"Created {len(config.lag_features)} lag features")
            
            # Add rolling features
            if config.rolling_window_sizes:
                rolling_df = self.feature_engineer.create_rolling_features(
                    data, 
                    config.rolling_window_sizes
                )
                df = pd.concat([df, rolling_df], axis=1)
                self.logger.info(
                    f"Created rolling features for {len(config.rolling_window_sizes)} windows"
                )
            
            # Add time-based features
            time_df = self.feature_engineer.create_time_features(data)
            df = pd.concat([df, time_df], axis=1)
            self.logger.info("Created time-based features")
            
            # Add OHLCV-specific features
            ohlcv_features = self._create_ohlcv_features(data)
            df = pd.concat([df, ohlcv_features], axis=1)
            self.logger.info("Created OHLCV-specific features")
            
            self.logger.info(f"Feature creation completed: {len(df.columns)} total features")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature creation failed for {series_id}", e)
            raise
    
    def validate_data(self, series_id: str) -> dict:
        """
        Validate time series data quality.
        
        Returns:
            Dictionary with validation metrics
        """
        try:
            data = self.repository.get_raw_data(series_id)
            df = data.to_dataframe()
            
            # Calculate statistics for each OHLCV column
            ohlcv_stats = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                ohlcv_stats[col] = {
                    'missing_count': int(df[col].isna().sum()),
                    'missing_percentage': float((df[col].isna().sum() / len(df)) * 100),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            
            validation = {
                'total_points': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'ohlcv_stats': ohlcv_stats,
                'data_quality_checks': {
                    'high_ge_low': int((df['high'] >= df['low']).sum()),
                    'high_ge_open': int((df['high'] >= df['open']).sum()),
                    'high_ge_close': int((df['high'] >= df['close']).sum()),
                    'low_le_open': int((df['low'] <= df['open']).sum()),
                    'low_le_close': int((df['low'] <= df['close']).sum()),
                    'volume_positive': int((df['volume'] > 0).sum())
                }
            }
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Validation failed for {series_id}", e)
            raise