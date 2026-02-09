export interface PredictionRequest {
  firm_size: number;
  industry: string;
  operational_maturity: number;
  ai_usage_level: string;
  ai_investment: number;
  primary_use_case: string;
}

export interface PredictionResponse {
  predicted_roi: number;
  direction: 'positive' | 'neutral' | 'negative';
  interpretation: string;
  model_version: string;
  timestamp: string;
}

export async function fetchPrediction(data: PredictionRequest): Promise<PredictionResponse> {
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const baseRoi = (data.ai_investment / data.firm_size) * 0.15;
  const maturityBonus = data.operational_maturity * 2;
  const usageMultiplier = 
    data.ai_usage_level === 'High' ? 1.5 :
    data.ai_usage_level === 'Moderate' ? 1.2 :
    data.ai_usage_level === 'Low' ? 0.8 : 0.5;
  
  const predicted_roi = (baseRoi + maturityBonus) * usageMultiplier + (Math.random() * 10 - 5);

  const direction: 'positive' | 'neutral' | 'negative' = 
    predicted_roi > 10 ? 'positive' :
    predicted_roi > 0 ? 'neutral' : 'negative';

  const interpretation = 
    direction === 'positive'
      ? `Based on your inputs, the model predicts a positive ROI of ${predicted_roi.toFixed(1)}%. This suggests that AI adoption may be beneficial for your firm profile. Firms with similar characteristics and ${data.ai_usage_level.toLowerCase()} AI usage levels have historically seen favorable returns on their AI investments.`
      : direction === 'neutral'
      ? `Based on your inputs, the model predicts a neutral ROI of ${predicted_roi.toFixed(1)}%. This suggests that AI adoption may break even for your firm profile. Consider optimizing operational maturity and AI usage strategies to improve potential returns.`
      : `Based on your inputs, the model predicts a negative ROI of ${predicted_roi.toFixed(1)}%. This suggests that AI adoption may not be immediately beneficial for your firm profile. Consider building foundational capabilities before significant AI investment.`;

  return {
    predicted_roi,
    direction,
    interpretation,
    model_version: 'v1.0.0-mock',
    timestamp: new Date().toISOString(),
  };
}
