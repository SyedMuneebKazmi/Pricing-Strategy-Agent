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

# Configuration for production
class ProductionConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'prod-key-456')
    DEBUG = False
    DATA_PATH = 'data/sample_data.csv'
    MODEL_PATH = 'models/pricing_model.pkl'
    PROCESSOR_PATH = 'models/data_processor.pkl'
    DEFAULT_MIN_MARGIN = 0.20
    DEFAULT_MAX_PRICE_MULTIPLIER = 2.5
    DEMAND_LEVELS = {'Low': 0, 'Medium': 1, 'High': 2}
    SEASONALITY_LEVELS = {'Low': 0, 'Normal': 1, 'Peak': 2}

app = Flask(__name__)
app.config.from_object(ProductionConfig)

# Global variables
model = None
data_processor = None
pricing_agent = None

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

class SimplePricingAgent:
    def __init__(self, config):
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

# Initialize the agent
pricing_agent = SimplePricingAgent(ProductionConfig())

@app.route('/')
def index():
    return render_template('index.html', agent_status="ready", model_loaded=False)

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/recommend', methods=['POST'])
def recommend_price():
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
        
        # Validate
        if input_data['cost_price'] <= 0:
            return jsonify({
                'error': 'Cost price must be greater than 0',
                'status_code': 'ERROR_VALIDATION'
            }), 400
        
        # Set defaults
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
    return render_template('about.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'agent_type': 'simple',
        'version': '1.0.0'
    })

@app.route('/api/features')
def feature_info():
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
        ]
    }
    return jsonify(features_info)

if __name__ == '__main__':
    app.run(debug=False)