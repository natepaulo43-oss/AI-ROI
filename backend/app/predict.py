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
    
    # Generate monthly forecast based on deployment timeline with gradual ramp-up
    # Calculate number of months from days_to_deployment
    forecast_month_count = max(12, int(np.ceil(input_data.days_to_deployment / 30)))
    
    # Dynamic ramp-up parameters based on use case complexity
    # Based on industry research: customer service (3-6mo), predictive maintenance (12-24mo)
    use_case_maturation = {
        # Fast adoption (3-5 months) - Simple, immediate value
        'customer service bot': {'ramp_months': 4, 'initial': 0.45},
        'chatbot': {'ramp_months': 4, 'initial': 0.45},
        'document processing': {'ramp_months': 5, 'initial': 0.40},
        'process automation': {'ramp_months': 5, 'initial': 0.40},
        
        # Medium adoption (6-8 months) - Moderate complexity
        'quality control vision': {'ramp_months': 6, 'initial': 0.35},
        'fraud detection': {'ramp_months': 7, 'initial': 0.30},
        'personalization engine': {'ramp_months': 6, 'initial': 0.35},
        'sentiment analysis': {'ramp_months': 6, 'initial': 0.35},
        'sales automation': {'ramp_months': 7, 'initial': 0.32},
        'pricing optimization': {'ramp_months': 7, 'initial': 0.32},
        
        # Slow adoption (8-10 months) - High complexity, data-intensive
        'predictive analytics': {'ramp_months': 8, 'initial': 0.30},
        'demand forecasting': {'ramp_months': 8, 'initial': 0.30},
        'supply chain optimization': {'ramp_months': 8, 'initial': 0.28},
        'inventory management': {'ramp_months': 8, 'initial': 0.28},
        'risk assessment': {'ramp_months': 9, 'initial': 0.25}
    }
    
    # Get maturation profile for the use case
    maturation = use_case_maturation.get(
        input_data.ai_use_case,
        {'ramp_months': 6, 'initial': 0.35, 'growth_rate': 0.02}  # default
    )
    
    # Adjust ramp-up based on company size (larger = slower adoption)
    size_multiplier = {
        'micro': 0.9,
        'pequena': 1.0,
        'media': 1.15,
        'grande': 1.3
    }.get(input_data.company_size, 1.0)
    
    # Adjust based on deployment type
    deployment_multiplier = {
        'cloud': 0.95,
        'on_premise': 1.1,
        'hybrid': 1.2
    }.get(input_data.deployment_type, 1.0)
    
    # Calculate effective ramp months
    effective_ramp_months = maturation['ramp_months'] * size_multiplier * deployment_multiplier
    
    forecast_months = []
    for month in range(1, forecast_month_count + 1):
        if month <= effective_ramp_months:
            # Ramp-up phase: sigmoid curve for smooth acceleration
            progress = month / effective_ramp_months
            ramp_factor = maturation['initial'] + (1 - maturation['initial']) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        else:
            # Post-ramp stabilization: model's prediction represents stabilized ROI
            # Add small realistic variation (±2-5%) to show natural performance fluctuation
            months_after_ramp = month - effective_ramp_months
            # Diminishing variation over time (more stable as system matures)
            variation_amplitude = 0.05 * np.exp(-0.1 * months_after_ramp)
            # Use deterministic variation based on month (not random) for consistent forecasts
            variation = variation_amplitude * np.sin(months_after_ramp * 0.5)
            ramp_factor = 1.0 + variation
        
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
