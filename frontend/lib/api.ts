// Currency conversion rate (approximate)
const USD_TO_EUR = 0.91; // 1 USD = 0.91 EUR
const EUR_TO_USD = 1.10; // 1 EUR = 1.10 USD

// Backend API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PredictionRequest {
  // Company characteristics
  year: number;
  quarter: string;
  sector: string;
  company_size: string;
  revenue_m_usd: number; // Frontend uses USD
  
  // AI deployment details
  ai_use_case: string;
  deployment_type: string;
  days_diagnostic: number;
  days_poc: number;
  days_to_deployment: number;
  investment_usd: number; // Frontend uses USD
  
  // Early signals (optional)
  time_saved_hours_month: number;
  revenue_increase_percent: number;
  
  // Configuration
  human_in_loop: boolean;
}

export interface PredictionResponse {
  predicted_roi: number;
  direction: 'positive' | 'neutral' | 'negative';
  interpretation: string;
  model_version: string;
  confidence_note: string;
  timestamp: string;
}

// Backend expects EUR, so we convert from USD
interface BackendRequest {
  year: number;
  quarter: string;
  sector: string;
  company_size: string;
  revenue_m_eur: number;
  ai_use_case: string;
  deployment_type: string;
  days_diagnostic: number;
  days_poc: number;
  days_to_deployment: number;
  investment_eur: number;
  time_saved_hours_month: number;
  revenue_increase_percent: number;
  human_in_loop: number;
}

interface BackendResponse {
  predicted_roi: number;
  model_version: string;
  confidence_note: string;
}

export async function fetchPrediction(data: PredictionRequest): Promise<PredictionResponse> {
  // Convert USD to EUR for backend
  const backendRequest: BackendRequest = {
    year: data.year,
    quarter: data.quarter,
    sector: data.sector.toLowerCase(),
    company_size: data.company_size,
    revenue_m_eur: data.revenue_m_usd * USD_TO_EUR,
    ai_use_case: data.ai_use_case.toLowerCase(),
    deployment_type: data.deployment_type,
    days_diagnostic: data.days_diagnostic,
    days_poc: data.days_poc,
    days_to_deployment: data.days_to_deployment,
    investment_eur: data.investment_usd * USD_TO_EUR,
    time_saved_hours_month: data.time_saved_hours_month,
    revenue_increase_percent: data.revenue_increase_percent,
    human_in_loop: data.human_in_loop ? 1 : 0,
  };

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backendRequest),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    const backendResponse: BackendResponse = await response.json();
    const predicted_roi = backendResponse.predicted_roi;

    // Determine direction based on ROI
    const direction: 'positive' | 'neutral' | 'negative' = 
      predicted_roi > 100 ? 'positive' :
      predicted_roi > 0 ? 'neutral' : 'negative';

    // Generate interpretation
    const interpretation = generateInterpretation(predicted_roi, direction, data);

    return {
      predicted_roi,
      direction,
      interpretation,
      model_version: backendResponse.model_version,
      confidence_note: backendResponse.confidence_note,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Prediction API error:', error);
    throw new Error(
      error instanceof Error 
        ? error.message 
        : 'Failed to get prediction. Please ensure the backend API is running.'
    );
  }
}

function generateInterpretation(
  roi: number, 
  direction: 'positive' | 'neutral' | 'negative',
  data: PredictionRequest
): string {
  const roiFormatted = roi.toFixed(1);
  const hasEarlySignals = data.time_saved_hours_month > 0 || data.revenue_increase_percent > 0;
  
  if (direction === 'positive') {
    return `Based on your deployment characteristics, the ML model predicts a strong ROI of ${roiFormatted}%. This suggests that your AI investment is likely to generate significant returns. ${hasEarlySignals ? 'Your early deployment signals (time savings and revenue impact) support this positive outlook.' : 'Consider tracking early deployment metrics to validate this prediction.'}`;
  } else if (direction === 'neutral') {
    return `Based on your deployment characteristics, the ML model predicts a moderate ROI of ${roiFormatted}%. This suggests your AI investment may break even or generate modest returns. ${hasEarlySignals ? 'Monitor your early deployment metrics closely to optimize outcomes.' : 'Focus on capturing time savings and revenue impact early to improve ROI.'}`;
  } else {
    return `Based on your deployment characteristics, the ML model predicts a challenging ROI of ${roiFormatted}%. This suggests potential difficulties in achieving positive returns. ${hasEarlySignals ? 'Despite some early signals, consider reassessing deployment strategy and timeline.' : 'Focus on quick wins and measurable outcomes to improve trajectory. Consider extending diagnostic and POC phases.'}`;
  }
}
