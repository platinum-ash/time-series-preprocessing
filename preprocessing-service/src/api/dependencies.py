"""
Dependency injection container for the API.
Configures and provides all service dependencies.
"""
import os
from src.domain.service import PreprocessingService
from src.adapters.repository import TimescaleDBRepository
from src.adapters.missing_values import MissingValueHandler
from src.adapters.outlier_detection import StatisticalOutlierDetector
from src.adapters.feature_engineering import FeatureEngineer
from src.adapters.resampling import Resampler
from src.adapters.logging import PythonLogger


def get_preprocessing_service() -> PreprocessingService:
    """
    Factory function to create PreprocessingService with all dependencies.
    """
    
    # Get configuration from environment
    db_connection = os.getenv(
        'DATABASE_URL', 
        'postgresql://localhost/timeseries'
    )
    
    # Initialize adapters
    repository = TimescaleDBRepository(db_connection)
    missing_handler = MissingValueHandler()
    outlier_detector = StatisticalOutlierDetector()
    feature_engineer = FeatureEngineer()
    resampler = Resampler()
    logger = PythonLogger("preprocessing-api")
    
    # Create and return service
    return PreprocessingService(
        repository=repository,
        missing_handler=missing_handler,
        outlier_detector=outlier_detector,
        feature_engineer=feature_engineer,
        resampler=resampler,
        logger=logger
    )


# Singleton instance for the application
_service_instance = None

def get_service() -> PreprocessingService:
    """Get or create singleton service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = get_preprocessing_service()
    return _service_instance