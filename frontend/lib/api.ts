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

export interface MonthlyForecast {
  month: number;
  roi: number;
  lower: number;
  upper: number;
}

export interface PredictionResponse {
  prediction: string; // "High" or "Not-High"
  probability_high: number; // 0-1
  probability_not_high: number; // 0-1
  confidence: number; // 0-1
  threshold: number; // 145.5%
  interpretation: string;
  direction: 'high' | 'not-high';
  timestamp: string;
  predicted_roi: number; // Predicted ROI percentage
  roi_lower_bound: number; // Lower confidence interval
  roi_upper_bound: number; // Upper confidence interval
  forecast_months: MonthlyForecast[]; // Monthly forecast data
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
  prediction: string; // "High" or "Not-High"
  probability_high: number;
  probability_not_high: number;
  confidence: number;
  threshold: number;
  interpretation: string;
  predicted_roi: number;
  roi_lower_bound: number;
  roi_upper_bound: number;
  forecast_months: MonthlyForecast[];
}

// Retry helper for handling cold starts on free hosting tiers
async function fetchWithRetry(
  url: string, 
  options: RequestInit, 
  maxRetries = 3,
  initialDelay = 2000
): Promise<Response> {
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      lastError = error as Error;
      
      // Don't retry on abort or if it's the last attempt
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout - the API may be warming up. Please try again.');
      }
      
      if (attempt < maxRetries - 1) {
        const delay = initialDelay * Math.pow(2, attempt);
        console.log(`Retry attempt ${attempt + 1}/${maxRetries} after ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError || new Error('Request failed after retries');
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
    const response = await fetchWithRetry(
      `${API_BASE_URL}/predict`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(backendRequest),
      },
      3, // max retries
      2000 // initial delay (2s)
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    const backendResponse: BackendResponse = await response.json();

    // Map backend response to frontend format
    const direction: 'high' | 'not-high' = 
      backendResponse.prediction === 'High' ? 'high' : 'not-high';

    return {
      prediction: backendResponse.prediction,
      probability_high: backendResponse.probability_high,
      probability_not_high: backendResponse.probability_not_high,
      confidence: backendResponse.confidence,
      threshold: backendResponse.threshold,
      interpretation: backendResponse.interpretation,
      direction,
      timestamp: new Date().toISOString(),
      predicted_roi: backendResponse.predicted_roi,
      roi_lower_bound: backendResponse.roi_lower_bound,
      roi_upper_bound: backendResponse.roi_upper_bound,
      forecast_months: backendResponse.forecast_months,
    };
  } catch (error) {
    console.error('Prediction API error:', error);
    throw new Error(
      error instanceof Error 
        ? error.message 
        : 'Failed to get prediction. The API may be warming up - please try again in a moment.'
    );
  }
}

// Legacy function - no longer used, interpretation comes from backend
// Kept for backward compatibility if needed
