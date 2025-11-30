import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_sample_model():
    """Create a simple model for Vercel deployment"""
    print("Creating sample model for Vercel...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'cost_price': np.random.uniform(10, 100, n_samples),
        'competitor_price': np.random.uniform(15, 150, n_samples),
        'demand_level_encoded': np.random.choice([0, 1, 2], n_samples),
        'seasonality_encoded': np.random.choice([0, 1, 2], n_samples),
    }
    
    # Create target (optimal price) based on features
    X = pd.DataFrame(data)
    y = (X['cost_price'] * 1.5 + X['competitor_price'] * 0.3 + 
         X['demand_level_encoded'] * 5 + X['seasonality_encoded'] * 3)
    
    # Add some noise
    y += np.random.normal(0, 5, n_samples)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/pricing_model.pkl')
    joblib.dump(scaler, 'models/data_processor.pkl')
    
    print("✅ Sample model created successfully!")
    print(f"Model saved to: models/pricing_model.pkl")
    print(f"Scaler saved to: models/data_processor.pkl")

if __name__ == '__main__':
    create_sample_model()