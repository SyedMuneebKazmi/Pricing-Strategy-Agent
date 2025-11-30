from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys
from pathlib import Path
import re
import csv

# Add the project root directory to Python path
# api/index.py -> api/ -> root
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Try to import custom modules, but provide fallbacks
try:
    from src.config import Config
    # from src.pricing_agent import PricingStrategyAgent
    print("✅ Config import successful")
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

# Initialize Flask with correct template and static folders relative to api/
app = Flask(__name__, template_folder='../templates', static_folder='../static')
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
    
    # Force fallback to SimplePricingAgent for now to avoid Vercel crash
    print("⚠️ Using simple pricing agent as fallback (forced)")
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
            
            print(f"Processing: cost=${cost_price}, competitor=${competitor_price}")
            
            # Smart pricing logic
            base_price = (cost_price * 0.4 + competitor_price * 0.6)
            
            # Adjust based on demand and seasonality
            demand_multiplier = {'Low': 0.85, 'Medium': 1.0, 'High': 1.15}
            seasonality_multiplier = {'Low': 0.9, 'Normal': 1.0, 'Peak': 1.1}
            
            adjusted_price = base_price * demand_multiplier[demand_level] * seasonality_multiplier[seasonality]
            
            # Apply business constraints
            min_price = cost_price * (1 + min_margin)
            constrained_price = max(min_price, min(max_price, adjusted_price))
            
            # Calculate business metrics
            base_demand = 100
            demand_factor = {'Low': 0.7, 'Medium': 1.0, 'High': 1.3}
            predicted_units = int(base_demand * demand_factor[demand_level])
            
            profit_per_unit = constrained_price - cost_price
            total_profit = profit_per_unit * predicted_units
            profit_margin = (profit_per_unit / constrained_price) * 100
            
            # Calculate confidence based on input quality
            confidence = 0.8
            if competitor_price > cost_price * 1.5:
                confidence += 0.1
            if min_margin <= 0.3:  # Reasonable margin
                confidence += 0.05
            
            confidence = min(0.95, confidence)
            
            return {
                'recommended_price': round(constrained_price, 2),
                'confidence_score': round(confidence, 2),
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
                    'min_margin': round(min_margin * 100, 1)
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
        
        if input_data['min_margin'] < 0:
            return jsonify({
                'error': 'Minimum margin cannot be negative',
                'status_code': 'ERROR_VALIDATION'
            }), 400
        
        # Set defaults
        if input_data['max_price'] == 0:
            input_data['max_price'] = input_data['cost_price'] * 2.5
        if input_data['competitor_price'] == 0:
            input_data['competitor_price'] = input_data['cost_price'] * 1.8
        
        # Get recommendation
        recommendation = pricing_agent.find_optimal_price(input_data)
        
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
        'app_initialized': app_initialized,
        'version': '2.0.0',
        'features': ['smart_pricing', 'demand_analysis', 'profit_optimization']
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
        },
        'pricing_strategy': 'AI-powered optimization with business constraints'
    }
    return jsonify(features_info)

@app.route('/examples')
def examples():
    """Provide example inputs"""
    examples_data = {
        'example_scenarios': [
            {
                'name': 'Electronics Product',
                'inputs': {
                    'cost_price': 25.00,
                    'competitor_price': 45.00,
                    'demand_level': 'Medium',
                    'seasonality': 'Normal',
                    'min_margin': 20
                }
            },
            {
                'name': 'High-End Fashion',
                'inputs': {
                    'cost_price': 50.00,
                    'competitor_price': 120.00,
                    'demand_level': 'High',
                    'seasonality': 'Peak',
                    'min_margin': 40
                }
            },
            {
                'name': 'Basic Commodity',
                'inputs': {
                    'cost_price': 5.00,
                    'competitor_price': 8.99,
                    'demand_level': 'Low',
                    'seasonality': 'Normal',
                    'min_margin': 15
                }
            }
        ]
    }
    return jsonify(examples_data)

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
