import os
from pathlib import Path

class Config:
    """Configuration settings for the Pricing Strategy Agent"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-123'
    DEBUG = False  # Important for production
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / 'data' / 'sample_data.csv'
    MODEL_PATH = BASE_DIR / 'models' / 'pricing_model.pkl'
    PROCESSOR_PATH = BASE_DIR / 'models' / 'data_processor.pkl'
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Business constraints
    DEFAULT_MIN_MARGIN = 0.20
    DEFAULT_MAX_PRICE_MULTIPLIER = 2.5
    
    # Feature columns
    FEATURE_COLUMNS = ['cost_price', 'competitor_price', 'demand_level_encoded', 'seasonality_encoded']
    TARGET_COLUMN = 'optimal_price'
    
    # Demand level mapping
    DEMAND_LEVELS = {'Low': 0, 'Medium': 1, 'High': 2}
    SEASONALITY_LEVELS = {'Low': 0, 'Normal': 1, 'Peak': 2}