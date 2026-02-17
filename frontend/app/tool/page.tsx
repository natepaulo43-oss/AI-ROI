'use client';

import { useState } from 'react';
import PageLayout from '@/components/layout/PageLayout';
import InputPanel from '@/components/tool/InputPanel';
import FormSection from '@/components/tool/FormSection';
import NumberInput from '@/components/tool/inputs/NumberInput';
import Dropdown from '@/components/tool/inputs/Dropdown';
import Slider from '@/components/tool/inputs/Slider';
import ResultsPanel from '@/components/tool/ResultsPanel';
import LoadingState from '@/components/tool/LoadingState';
import { fetchPrediction, PredictionResponse } from '@/lib/api';

const SECTOR_OPTIONS = [
  { value: 'agroalimentaire', label: 'Agriculture & Food' },
  { value: 'automotive', label: 'Automotive' },
  { value: 'construction', label: 'Construction' },
  { value: 'education', label: 'Education' },
  { value: 'energie', label: 'Energy & Utilities' },
  { value: 'finance', label: 'Finance & Banking' },
  { value: 'insurance', label: 'Insurance' },
  { value: 'logistique', label: 'Logistics & Supply Chain' },
  { value: 'manufacturing', label: 'Manufacturing' },
  { value: 'media', label: 'Media & Entertainment' },
  { value: 'pharma', label: 'Pharmaceutical' },
  { value: 'retail', label: 'Retail & E-commerce' },
  { value: 'sante', label: 'Healthcare' },
  { value: 'services pro', label: 'Professional Services' },
  { value: 'technology', label: 'Technology' },
  { value: 'telecom', label: 'Telecommunications' },
];

const USE_CASE_OPTIONS = [
  { value: 'customer service bot', label: 'Customer Service & Chatbots', group: 'Customer Experience' },
  { value: 'personalization engine', label: 'Personalization Engine', group: 'Customer Experience' },
  { value: 'sentiment analysis', label: 'Sentiment Analysis', group: 'Customer Experience' },
  
  { value: 'process automation', label: 'Process Automation (RPA)', group: 'Operations' },
  { value: 'quality control vision', label: 'Quality Control & Vision', group: 'Operations' },
  { value: 'document processing', label: 'Document Processing & OCR', group: 'Operations' },
  { value: 'inventory management', label: 'Inventory Management', group: 'Operations' },
  { value: 'supply chain optimization', label: 'Supply Chain Optimization', group: 'Operations' },
  
  { value: 'predictive analytics', label: 'Predictive Analytics', group: 'Analytics & Intelligence' },
  { value: 'demand forecasting', label: 'Demand Forecasting', group: 'Analytics & Intelligence' },
  { value: 'risk assessment', label: 'Risk Assessment', group: 'Analytics & Intelligence' },
  { value: 'fraud detection', label: 'Fraud Detection & Prevention', group: 'Analytics & Intelligence' },
  
  { value: 'sales automation', label: 'Sales & Marketing Automation', group: 'Revenue & Growth' },
  { value: 'pricing optimization', label: 'Pricing Optimization', group: 'Revenue & Growth' },
];

const COMPANY_SIZE_OPTIONS = [
  { value: 'pme', label: 'Small/Medium (<250 employees)' },
  { value: 'eti', label: 'Mid-sized (250-5,000 employees)' },
  { value: 'grande', label: 'Large Enterprise (>5,000 employees)' },
];

const DEPLOYMENT_TYPE_OPTIONS = [
  { value: 'analytics', label: 'Analytics / Data Platform' },
  { value: 'automation', label: 'Automation / RPA' },
  { value: 'hybrid', label: 'Hybrid (Cloud + On-premise)' },
  { value: 'nlp', label: 'NLP / Language Model' },
  { value: 'vision', label: 'Computer Vision' },
];

export default function Tool() {
  // Essential fields
  const [sector, setSector] = useState<string>('');
  const [companySize, setCompanySize] = useState<string>('');
  const [revenueUSD, setRevenueUSD] = useState<number>(0);
  const [aiUseCase, setAiUseCase] = useState<string>('');
  const [deploymentType, setDeploymentType] = useState<string>('');
  const [investmentUSD, setInvestmentUSD] = useState<number>(0);
  const [deploymentMonths, setDeploymentMonths] = useState<number>(0);

  // Optional - for better predictions
  const [timeSaved, setTimeSaved] = useState<number>(0);
  const [revenueIncrease, setRevenueIncrease] = useState<number>(0);

  // Hidden defaults
  const year = 2024;
  const quarter = 'q2';
  const humanInLoop = true;

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    // Validate required fields
    if (!sector || !companySize || !aiUseCase || !deploymentType) {
      setError('Please fill in all required fields (Industry, Company Size, Use Case, and Technology Type)');
      return;
    }

    if (revenueUSD <= 0) {
      setError('Please enter a valid annual revenue greater than $0');
      return;
    }

    if (investmentUSD <= 0) {
      setError('Please enter a valid investment budget greater than $0');
      return;
    }

    if (deploymentMonths <= 0) {
      setError('Please enter a valid timeline (at least 1 month)');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      // Calculate derived fields from deployment months
      const totalDays = deploymentMonths * 30;
      const daysDiagnostic = Math.round(totalDays * 0.15); // 15% diagnostic
      const daysPoc = Math.round(totalDays * 0.30); // 30% POC
      const daysToDeployment = totalDays;

      const prediction = await fetchPrediction({
        year,
        quarter,
        sector,
        company_size: companySize,
        revenue_m_usd: revenueUSD / 1000000, // Convert raw USD to millions
        ai_use_case: aiUseCase,
        deployment_type: deploymentType,
        days_diagnostic: daysDiagnostic,
        days_poc: daysPoc,
        days_to_deployment: daysToDeployment,
        investment_usd: investmentUSD, // Already in raw USD
        time_saved_hours_month: timeSaved,
        revenue_increase_percent: revenueIncrease,
        human_in_loop: humanInLoop,
      });
      setResult(prediction);
    } catch (error) {
      console.error('Prediction failed:', error);
      setError(error instanceof Error ? error.message : 'Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-12 py-16 min-h-screen">
      {/* Page header with asymmetric layout */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          AI ROI Calculator
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          Predict Your
          <br />
          AI Returns
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md leading-relaxed">
          Get a data-driven ROI prediction based on 500+ real AI implementations. Fill in your project details below.
        </p>
      </div>

      {/* Asymmetric grid: form left, results right */}
      <div className="grid grid-cols-12 gap-12">
        {/* Left column: Input form (5 columns) */}
        <div className="col-span-5">
          <InputPanel onSubmit={handleSubmit} isLoading={isLoading}>
            <FormSection title="Your Business">
              <Dropdown
                label="Industry Sector"
                value={sector}
                onChange={setSector}
                options={SECTOR_OPTIONS}
              />
              <Dropdown
                label="Company Size"
                value={companySize}
                onChange={setCompanySize}
                options={COMPANY_SIZE_OPTIONS}
              />
              <NumberInput
                label="Annual Revenue (USD)"
                value={revenueUSD}
                onChange={setRevenueUSD}
                step={100000}
                placeholder="e.g., 5000000 ($5M)"
              />
            </FormSection>

            <FormSection title="AI Project Details">
              <Dropdown
                label="Primary AI Use Case"
                value={aiUseCase}
                onChange={setAiUseCase}
                options={USE_CASE_OPTIONS}
              />
              <Dropdown
                label="Technology Type"
                value={deploymentType}
                onChange={setDeploymentType}
                options={DEPLOYMENT_TYPE_OPTIONS}
              />
              <NumberInput
                label="Total Investment Budget (USD)"
                value={investmentUSD}
                onChange={setInvestmentUSD}
                step={10000}
                placeholder="e.g., 250000 ($250K)"
              />
              <NumberInput
                label="Expected Timeline (Months)"
                value={deploymentMonths}
                onChange={setDeploymentMonths}
                step={1}
                placeholder="e.g., 6-12 months"
              />
            </FormSection>

            <FormSection title="Expected Benefits (Optional - Improves Accuracy)">
              <NumberInput
                label="Time Saved (Hours/Month)"
                value={timeSaved}
                onChange={setTimeSaved}
                step={10}
                placeholder="e.g., 200 hours/month"
              />
              <NumberInput
                label="Revenue Increase (%)"
                value={revenueIncrease}
                onChange={setRevenueIncrease}
                step={1}
                placeholder="e.g., 5% increase"
              />
            </FormSection>
          </InputPanel>
        </div>

        {/* Right column: Results (6 columns, offset) */}
        <div className="col-span-6 col-start-7">
          {error && (
            <div className="bg-red-900/20 border border-red-500/30 rounded-[2rem] p-8 mb-6">
              <p className="text-red-400 text-sm">{error}</p>
              <p className="text-red-300/60 text-xs mt-2">
                Make sure the backend API is running at http://localhost:8000
              </p>
            </div>
          )}
          {isLoading && <LoadingState />}
          {!isLoading && result && (
            <ResultsPanel
              predictedRoi={result.predicted_roi}
              direction={result.direction}
              interpretation={result.interpretation}
              forecastData={result.forecast_months}
              threshold={result.threshold}
              confidence={result.confidence}
            />
          )}
          {!isLoading && !result && !error && (
            <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-square flex items-center justify-center">
              <p className="text-[0.7rem] uppercase tracking-[0.15em] text-[#b8a894]">
                Results will appear here
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
