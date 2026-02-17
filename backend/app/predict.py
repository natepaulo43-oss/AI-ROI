import pandas as pd
import numpy as np
from .schemas import PredictionInput, PredictionOutput, MonthlyForecast

def make_prediction(models, input_data: PredictionInput) -> PredictionOutput:
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
    
    # Additional features for classifier
    input_df['revenue_investment_ratio'] = input_df['revenue_m_eur'] / (input_df['investment_eur'] / 1_000_000 + 1)
    input_df['time_efficiency'] = input_df['time_saved_hours_month'] / (input_df['total_prep_time'] + 1)
    input_df['revenue_time_interaction'] = input_df['revenue_increase_percent'] * input_df['time_saved_hours_month']
    
    # Select features for classifier (matches roi_api.py)
    classifier_features = [
        'log_investment', 'log_revenue', 'investment_per_day',
        'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
        'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year',
        'revenue_investment_ratio', 'time_efficiency', 'revenue_time_interaction',
        'sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter'
    ]
    
    X_classifier = input_df[classifier_features]
    
    # Select features for regression (original features)
    regression_features = [
        'log_investment', 'log_revenue', 'investment_ratio',
        'investment_per_day', 'diagnostic_efficiency', 'poc_efficiency',
        'total_prep_time', 'deployment_speed', 'size_investment_interaction',
        'is_large_company', 'is_hybrid_deployment', 'human_in_loop', 'year',
        'time_saved_hours_month', 'revenue_increase_percent',
        'has_revenue_increase', 'has_time_savings',
        'sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter'
    ]
    
    X_regression = input_df[regression_features]
    
    # Get classification prediction and probabilities
    classifier = models['classifier']
    regression = models['regression']
    
    prediction_binary = classifier.predict(X_classifier)[0]  # 0 = Not-High, 1 = High
    probabilities = classifier.predict_proba(X_classifier)[0]  # [prob_not_high, prob_high]
    
    # Get continuous ROI prediction from regression model
    predicted_roi = float(regression.predict(X_regression)[0])
    
    # Calculate confidence intervals (using MAE of 62.67% from model documentation)
    mae = 62.67
    roi_lower = predicted_roi - mae
    roi_upper = predicted_roi + mae
    
    # Generate monthly forecast (12 months) with gradual ramp-up
    forecast_months = []
    for month in range(1, 13):
        # Ramp-up curve: starts at 30% of predicted, reaches 100% by month 6, then stabilizes
        if month <= 6:
            ramp_factor = 0.3 + (0.7 * (month / 6))
        else:
            ramp_factor = 1.0 + (0.1 * np.random.normal(0, 0.1))  # Small variation after stabilization
        
        month_roi = predicted_roi * ramp_factor
        month_lower = roi_lower * ramp_factor
        month_upper = roi_upper * ramp_factor
        
        forecast_months.append(MonthlyForecast(
            month=month,
            roi=round(month_roi, 2),
            lower=round(month_lower, 2),
            upper=round(month_upper, 2)
        ))
    
    prob_not_high = float(probabilities[0])
    prob_high = float(probabilities[1])
    confidence = max(prob_high, prob_not_high)
    
    # Interpret prediction
    if prediction_binary == 1:
        prediction_label = "High"
        interpretation = f"High ROI Expected (≥145.5%). Confidence: {prob_high*100:.1f}%"
    else:
        prediction_label = "Not-High"
        interpretation = f"Not-High ROI Expected (<145.5%). Confidence: {prob_not_high*100:.1f}%"
    
    # Create complete output
    return PredictionOutput(
        prediction=prediction_label,
        probability_high=prob_high,
        probability_not_high=prob_not_high,
        confidence=confidence,
        threshold=145.5,
        interpretation=interpretation,
        predicted_roi=round(predicted_roi, 2),
        roi_lower_bound=round(roi_lower, 2),
        roi_upper_bound=round(roi_upper, 2),
        forecast_months=forecast_months
    )
