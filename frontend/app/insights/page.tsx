import PageLayout from '@/components/layout/PageLayout';
import ModelCard from '@/components/insights/ModelCard';
import FeatureImportanceChart from '@/components/insights/FeatureImportanceChart';
import TechnicalSpecs from '@/components/insights/TechnicalSpecs';

const MOCK_FEATURE_IMPORTANCE = [
  { feature: 'AI Investment Amount', importance: 0.35 },
  { feature: 'Operational Maturity', importance: 0.28 },
  { feature: 'Current AI Usage Level', importance: 0.18 },
  { feature: 'Firm Size', importance: 0.12 },
  { feature: 'Industry Sector', importance: 0.05 },
  { feature: 'Primary Use Case', importance: 0.02 },
];

const MOCK_METRICS = [
  {
    label: 'R² Score',
    value: '0.82',
    description: 'Proportion of variance explained by the model',
  },
  {
    label: 'RMSE',
    value: '4.3%',
    description: 'Root mean squared error of predictions',
  },
  {
    label: 'MAE',
    value: '3.1%',
    description: 'Mean absolute error across test set',
  },
  {
    label: 'Training Samples',
    value: '1,247',
    description: 'Number of SME cases used for training',
  },
];

export default function Insights() {
  return (
    <div className="mx-auto max-w-7xl px-12 py-16">
      {/* Page header */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          Model Analysis
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          Insights
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Understanding the machine learning model behind ROI predictions
        </p>
      </div>

      {/* Asymmetric content sections */}
      <div className="space-y-32">
        {/* Model overview with visual element */}
        <div className="grid grid-cols-12 gap-12 items-start">
          <div className="col-span-5">
            <ModelCard />
          </div>
          <div className="col-span-6 col-start-7">
            <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-[4/3] flex items-center justify-center p-12">
              <div className="text-center">
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-4">
                  XGBoost Architecture
                </div>
                <p className="text-sm text-[#e8dfd5] leading-relaxed font-light">
                  Gradient boosting ensemble with sequential tree construction for optimal prediction accuracy
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Feature importance - full width */}
        <FeatureImportanceChart data={MOCK_FEATURE_IMPORTANCE} />

        {/* Technical specs with visual balance */}
        <div className="grid grid-cols-12 gap-12">
          <div className="col-span-6">
            <TechnicalSpecs metrics={MOCK_METRICS} />
          </div>
          <div className="col-span-5 col-start-8">
            <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">
              Note on Transparency
            </div>
            <p className="text-sm text-[#e8dfd5] leading-relaxed font-light">
              This model was trained on historical data from SME AI adoption initiatives. Feature 
              importance values help explain which factors most strongly influence predicted ROI. 
              The model achieves strong performance (R² = 0.82) while maintaining interpretability, 
              making it suitable for academic research and decision-support applications.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
