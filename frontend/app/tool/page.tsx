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

const INDUSTRY_OPTIONS = [
  { value: 'Manufacturing', label: 'Manufacturing' },
  { value: 'Retail', label: 'Retail' },
  { value: 'Healthcare', label: 'Healthcare' },
  { value: 'Finance', label: 'Finance' },
  { value: 'Technology', label: 'Technology' },
  { value: 'Professional Services', label: 'Professional Services' },
  { value: 'Other', label: 'Other' },
];

const AI_USAGE_OPTIONS = [
  { value: 'None', label: 'None' },
  { value: 'Low', label: 'Low' },
  { value: 'Moderate', label: 'Moderate' },
  { value: 'High', label: 'High' },
];

const USE_CASE_OPTIONS = [
  { value: 'Operations', label: 'Operations' },
  { value: 'Finance', label: 'Finance' },
  { value: 'Marketing', label: 'Marketing' },
  { value: 'HR', label: 'Human Resources' },
  { value: 'Customer Service', label: 'Customer Service' },
  { value: 'Other', label: 'Other' },
];

export default function Tool() {
  const [firmSize, setFirmSize] = useState<number>(100);
  const [industry, setIndustry] = useState<string>('');
  const [operationalMaturity, setOperationalMaturity] = useState<number>(3);
  const [aiUsageLevel, setAiUsageLevel] = useState<string>('');
  const [aiInvestment, setAiInvestment] = useState<number>(50000);
  const [primaryUseCase, setPrimaryUseCase] = useState<string>('');

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleSubmit = async () => {
    setIsLoading(true);
    setResult(null);

    try {
      const prediction = await fetchPrediction({
        firm_size: firmSize,
        industry,
        operational_maturity: operationalMaturity,
        ai_usage_level: aiUsageLevel,
        ai_investment: aiInvestment,
        primary_use_case: primaryUseCase,
      });
      setResult(prediction);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-12 py-16 min-h-screen">
      {/* Page header with asymmetric layout */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          Prediction Tool
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          Calculate
          <br />
          ROI
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Input your firm characteristics and AI adoption parameters
        </p>
      </div>

      {/* Asymmetric grid: form left, results right */}
      <div className="grid grid-cols-12 gap-12">
        {/* Left column: Input form (5 columns) */}
        <div className="col-span-5">
          <InputPanel onSubmit={handleSubmit} isLoading={isLoading}>
            <FormSection title="Firm Characteristics">
              <NumberInput
                label="Number of Employees"
                value={firmSize}
                onChange={setFirmSize}
                min={1}
                max={10000}
                placeholder="150"
              />
              <Dropdown
                label="Industry"
                value={industry}
                onChange={setIndustry}
                options={INDUSTRY_OPTIONS}
              />
              <Slider
                label="Operational Maturity"
                value={operationalMaturity}
                onChange={setOperationalMaturity}
                min={1}
                max={5}
              />
            </FormSection>

            <FormSection title="AI Adoption">
              <Dropdown
                label="Current AI Usage Level"
                value={aiUsageLevel}
                onChange={setAiUsageLevel}
                options={AI_USAGE_OPTIONS}
              />
              <NumberInput
                label="AI Investment (USD)"
                value={aiInvestment}
                onChange={setAiInvestment}
                min={0}
                step={1000}
                placeholder="50000"
              />
              <Dropdown
                label="Primary Use Case"
                value={primaryUseCase}
                onChange={setPrimaryUseCase}
                options={USE_CASE_OPTIONS}
              />
            </FormSection>
          </InputPanel>
        </div>

        {/* Right column: Results (6 columns, offset) */}
        <div className="col-span-6 col-start-7">
          {isLoading && <LoadingState />}
          {!isLoading && result && (
            <ResultsPanel
              predictedRoi={result.predicted_roi}
              direction={result.direction}
              interpretation={result.interpretation}
            />
          )}
          {!isLoading && !result && (
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
