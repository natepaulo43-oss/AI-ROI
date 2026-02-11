import pandas as pd
import numpy as np
from .schemas import PredictionInput, PredictionOutput

def make_prediction(model, input_data: PredictionInput) -> PredictionOutput:
    """
    Make ROI prediction using the trained model.
    Converts input data to the format expected by the model pipeline.
    Applies the same feature engineering as during training.
    """
    
    # Convert input to DataFrame with raw features
    input_df = pd.DataFrame([{
        'year': input_data.year,
        'quarter': input_data.quarter,
        'sector': input_data.sector,
        'company_size': input_data.company_size,
        'revenue_m_eur': input_data.revenue_m_eur,
        'ai_use_case': input_data.ai_use_case,
        'deployment_type': input_data.deployment_type,
        'days_diagnostic': input_data.days_diagnostic,
        'days_poc': input_data.days_poc,
        'days_to_deployment': input_data.days_to_deployment,
        'investment_eur': input_data.investment_eur,
        'time_saved_hours_month': input_data.time_saved_hours_month,
        'revenue_increase_percent': input_data.revenue_increase_percent,
        'human_in_loop': input_data.human_in_loop
    }])
    
    # Apply feature engineering (same as training)
    # Log transforms
    input_df['log_investment'] = np.log1p(input_df['investment_eur'])
    input_df['log_revenue'] = np.log1p(input_df['revenue_m_eur'])
    
    # Ratios and efficiency metrics
    input_df['investment_ratio'] = input_df['investment_eur'] / (input_df['revenue_m_eur'] * 1_000_000)
    input_df['investment_per_day'] = input_df['investment_eur'] / (input_df['days_to_deployment'] + 1)
    input_df['diagnostic_efficiency'] = input_df['days_diagnostic'] / (input_df['days_to_deployment'] + 1)
    input_df['poc_efficiency'] = input_df['days_poc'] / (input_df['days_to_deployment'] + 1)
    
    # Time-based features
    input_df['total_prep_time'] = input_df['days_diagnostic'] + input_df['days_poc']
    input_df['deployment_speed'] = 1 / (input_df['days_to_deployment'] + 1)
    
    # Interaction features
    input_df['size_investment_interaction'] = input_df['log_revenue'] * input_df['log_investment']
    
    # Binary flags
    input_df['is_large_company'] = (input_df['company_size'] == 'grande').astype(int)
    input_df['is_hybrid_deployment'] = (input_df['deployment_type'] == 'hybrid').astype(int)
    input_df['human_in_loop'] = input_df['human_in_loop'].astype(int)
    input_df['has_revenue_increase'] = (input_df['revenue_increase_percent'] > 0).astype(int)
    input_df['has_time_savings'] = (input_df['time_saved_hours_month'] > 0).astype(int)
    
    # Select features in the same order as training
    numeric_features = [
        'log_investment', 'log_revenue', 'investment_ratio',
        'investment_per_day', 'diagnostic_efficiency', 'poc_efficiency',
        'total_prep_time', 'deployment_speed', 'size_investment_interaction',
        'is_large_company', 'is_hybrid_deployment', 'human_in_loop', 'year',
        'time_saved_hours_month', 'revenue_increase_percent',
        'has_revenue_increase', 'has_time_savings'
    ]
    categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']
    
    X = input_df[numeric_features + categorical_features]
    
    # Make prediction
    prediction = model.predict(X)
    predicted_roi = float(prediction[0])
    
    # Create output with metadata
    return PredictionOutput(
        predicted_roi=round(predicted_roi, 2),
        model_version="v2.0_practical",
        confidence_note="Moderate confidence (R²=0.42). Average error ±63%. Best used with early deployment signals."
    )
