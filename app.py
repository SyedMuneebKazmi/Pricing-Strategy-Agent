from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Try to import custom modules, but provide fallbacks
try:
    from config.config import Config
    from src.pricing_agent import PricingStrategyAgent
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback configuration
    class Config:
        DATA_PATH = 'data/sample_data.csv'
        MODEL_PATH = 'models/pricing_model.pkl'
        PROCESSOR_PATH = 'models/data_processor.pkl'
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        FEATURE_COLUMNS = ['cost_price', 'competitor_price', 'demand_level_encoded', 'seasonality_encoded']
        TARGET_COLUMN = 'optimal_price'
        DEMAND_LEVELS = {'Low': 0, 'Medium': 1, 'High': 2}
        SEASONALITY_LEVELS = {'Low': 0, 'Normal': 1, 'Peak': 2}
        DEFAULT_MIN_MARGIN = 0.20
        DEFAULT_MAX_PRICE_MULTIPLIER = 2.5
        SECRET_KEY = 'dev-key-123'
        DEBUG = False  # Important: Set to False for production

app = Flask(__name__)
app.config['SECRET_KEY'] = getattr(Config, 'SECRET_KEY', 'dev-key-123')
app.config['DEBUG'] = False  # Disable debug mode in production

# Global variables for model and processor
model = None
data_processor = None
pricing_agent = None
app_initialized = False

def clean_number_input(value):
    """Clean and convert number inputs safely"""
    if not value or value == '':
        return 0.0
    
    value_str = str(value).strip()
    value_str = re.sub(r'[$,]', '', value_str)
    
    try:
        return float(value_str)
    except ValueError:
        return 0.0

def load_model():
    """Load the trained model and data processor"""
    global model, data_processor, pricing_agent, app_initialized
    
    try:
        config = Config()
        
        # Use absolute paths for Vercel
        model_path = current_dir / 'models' / 'pricing_model.pkl'
        processor_path = current_dir / 'models' / 'data_processor.pkl'
        
        print(f"🔍 Looking for model at: {model_path}")
        
        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Model file not found at {model_path}. Using simple pricing agent.")
            return False
        
        # Load data processor (scaler)
        if os.path.exists(processor_path):
            data_processor = joblib.load(processor_path)
            print("✅ Data processor loaded successfully")
        else:
            print(f"❌ Data processor not found at {processor_path}. Using simple pricing agent.")
            return False
        
        # Initialize pricing agent
        pricing_agent = PricingStrategyAgent(model, data_processor, config)
        print("✅ Pricing Strategy Agent initialized successfully")
        
        app_initialized = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Simple fallback pricing agent
class SimplePricingAgent:
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
    
    def find_optimal_price(self, input_data):
        try:
            cost_price = input_data['cost_price']
            competitor_price = input_data.get('competitor_price', cost_price * 1.8)
            demand_level = input_data.get('demand_level', 'Medium')
            seasonality = input_data.get('seasonality', 'Normal')
            min_margin = input_data.get('min_margin', 20) / 100.0
            max_price = input_data.get('max_price', cost_price * 2.5)
            
            # Pricing logic
            base_price = (cost_price + competitor_price) / 2
            demand_multiplier = {'Low': 0.9, 'Medium': 1.0, 'High': 1.1}
            seasonality_multiplier = {'Low': 0.9, 'Normal': 1.0, 'Peak': 1.1}
            
            adjusted_price = base_price * demand_multiplier[demand_level] * seasonality_multiplier[seasonality]
            
            # Apply constraints
            min_price = cost_price * (1 + min_margin)
            constrained_price = max(min_price, min(max_price, adjusted_price))
            
            # Calculate metrics
            predicted_units = 150
            profit_per_unit = constrained_price - cost_price
            total_profit = profit_per_unit * predicted_units
            profit_margin = (profit_per_unit / constrained_price) * 100
            
            return {
                'recommended_price': round(constrained_price, 2),
                'confidence_score': 0.85,
                'predicted_units_sold': predicted_units,
                'status_code': 'SUCCESS',
                'business_metrics': {
                    'profit_per_unit': round(profit_per_unit, 2),
                    'total_profit': round(total_profit, 2),
                    'profit_margin': round(profit_margin, 2)
                },
                'constraints_applied': {
                    'min_price': round(min_price, 2),
                    'max_price': round(max_price, 2),
                    'min_margin': min_margin * 100
                },
                'input_data': input_data
            }
            
        except Exception as e:
            return {
                'recommended_price': 0.0,
                'confidence_score': 0.0,
                'predicted_units_sold': 0,
                'status_code': f'ERROR: {str(e)}',
                'business_metrics': {},
                'constraints_applied': {},
                'input_data': input_data
            }

def initialize_app():
    """Initialize the application"""
    global pricing_agent, app_initialized
    print("🚀 Initializing Pricing Strategy Agent...")
    if not load_model():
        print("⚠️ Using simple pricing agent as fallback")
        pricing_agent = SimplePricingAgent(None, None, Config)
        app_initialized = True

@app.before_request
def before_request():
    """Initialize app on first request if not already initialized"""
    global app_initialized
    if not app_initialized:
        initialize_app()

@app.route('/')
def index():
    """Home page"""
    agent_status = "ready" if pricing_agent else "initializing"
    model_loaded = model is not None
    return render_template('index.html', agent_status=agent_status, model_loaded=model_loaded)

@app.route('/pricing')
def pricing():
    """Pricing recommendation page"""
    return render_template('pricing.html')

@app.route('/recommend', methods=['POST'])
def recommend_price():
    """API endpoint for price recommendations"""
    if pricing_agent is None:
        return jsonify({
            'error': 'Pricing agent not initialized',
            'status_code': 'ERROR_SYSTEM'
        }), 500
    
    try:
        # Get and clean input data
        cost_price = clean_number_input(request.form.get('cost_price', 0))
        competitor_price = clean_number_input(request.form.get('competitor_price', 0))
        min_margin = clean_number_input(request.form.get('min_margin', 20))
        max_price = clean_number_input(request.form.get('max_price', 0))
        
        input_data = {
            'product_id': request.form.get('product_id', '').strip(),
            'cost_price': cost_price,
            'competitor_price': competitor_price,
            'demand_level': request.form.get('demand_level', 'Medium'),
            'seasonality': request.form.get('seasonality', 'Normal'),
            'min_margin': min_margin,
            'max_price': max_price
        }
        
        # Validate required fields
        if input_data['cost_price'] <= 0:
            return jsonify({
                'error': 'Cost price must be greater than 0',
                'status_code': 'ERROR_VALIDATION'
            }), 400
        
        # Set defaults if not provided
        if input_data['max_price'] == 0:
            input_data['max_price'] = input_data['cost_price'] * 2.5
        if input_data['competitor_price'] == 0:
            input_data['competitor_price'] = input_data['cost_price'] * 1.8
        
        # Get recommendation
        recommendation = pricing_agent.find_optimal_price(input_data)
        recommendation['input_data'] = input_data
        
        return jsonify(recommendation)
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'status_code': 'ERROR_PROCESSING'
        }), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = 'healthy' if pricing_agent is not None else 'unhealthy'
    model_loaded = model is not None
    processor_loaded = data_processor is not None
    
    return jsonify({
        'status': status,
        'model_loaded': model_loaded,
        'processor_loaded': processor_loaded,
        'agent_type': 'advanced' if model_loaded else 'simple',
        'app_initialized': app_initialized
    })

@app.route('/api/features')
def feature_info():
    """API endpoint to get feature information"""
    features_info = {
        'demand_levels': [
            {'value': 'Low', 'description': 'Low market demand'},
            {'value': 'Medium', 'description': 'Average market demand'},
            {'value': 'High', 'description': 'High market demand'}
        ],
        'seasonality': [
            {'value': 'Low', 'description': 'Off-season or low season'},
            {'value': 'Normal', 'description': 'Regular season'},
            {'value': 'Peak', 'description': 'High season or peak period'}
        ],
        'default_values': {
            'min_margin': 0.2,
            'max_price_multiplier': 2.5
        }
    }
    return jsonify(features_info)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'status_code': 'ERROR_SERVER'
    }), 500

# Initialize the app
print("🚀 Starting Pricing Strategy Agent Application...")
initialize_app()

# Vercel requires this
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)