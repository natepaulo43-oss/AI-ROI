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
  { value: 'Manufacturing', label: 'Manufacturing' },
  { value: 'Finance', label: 'Finance' },
  { value: 'Retail', label: 'Retail' },
  { value: 'Services Pro', label: 'Professional Services' },
  { value: 'Sante', label: 'Healthcare' },
  { value: 'Telecom', label: 'Technology/Telecom' },
];

const USE_CASE_OPTIONS = [
  { value: 'Customer Service Bot', label: 'Customer Service Automation' },
  { value: 'Process Automation', label: 'Process Automation' },
  { value: 'Predictive Analytics', label: 'Predictive Analytics' },
  { value: 'Sales Automation', label: 'Sales & Marketing Automation' },
  { value: 'Document Processing', label: 'Document Processing' },
];

const COMPANY_SIZE_OPTIONS = [
  { value: 'pme', label: 'Small/Medium (< €50M revenue)' },
  { value: 'eti', label: 'Mid-sized (€50M - €1.5B)' },
  { value: 'grande', label: 'Large Enterprise (> €1.5B)' },
];

const DEPLOYMENT_TYPE_OPTIONS = [
  { value: 'hybrid', label: 'Hybrid (Cloud + On-premise)' },
  { value: 'analytics', label: 'Analytics/Data Platform' },
  { value: 'nlp', label: 'NLP/Language Model' },
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
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Get an AI-powered ROI prediction in under 2 minutes. Answer a few questions about your business and AI project.
        </p>
      </div>

      {/* Asymmetric grid: form left, results right */}
      <div className="grid grid-cols-12 gap-12">
        {/* Left column: Input form (5 columns) */}
        <div className="col-span-5">
          <InputPanel onSubmit={handleSubmit} isLoading={isLoading}>
            <FormSection title="Your Business">
              <Dropdown
                label="Industry"
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
                label="Annual Revenue"
                value={revenueUSD}
                onChange={setRevenueUSD}
                step={1000}
                placeholder="5000000"
              />
            </FormSection>

            <FormSection title="AI Project">
              <Dropdown
                label="What will AI help with?"
                value={aiUseCase}
                onChange={setAiUseCase}
                options={USE_CASE_OPTIONS}
              />
              <Dropdown
                label="Deployment Type"
                value={deploymentType}
                onChange={setDeploymentType}
                options={DEPLOYMENT_TYPE_OPTIONS}
              />
              <NumberInput
                label="Investment Budget"
                value={investmentUSD}
                onChange={setInvestmentUSD}
                step={1000}
                placeholder="50000"
              />
              <NumberInput
                label="Expected Timeline (Months)"
                value={deploymentMonths}
                onChange={setDeploymentMonths}
                placeholder="6"
              />
            </FormSection>

            <FormSection title="Early Results (Optional - Improves Accuracy)">
              <NumberInput
                label="Time Saved (Hours/Month)"
                value={timeSaved}
                onChange={setTimeSaved}
                placeholder="0"
              />
              <NumberInput
                label="Revenue Increase (%)"
                value={revenueIncrease}
                onChange={setRevenueIncrease}
                step={0.5}
                placeholder="0"
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
