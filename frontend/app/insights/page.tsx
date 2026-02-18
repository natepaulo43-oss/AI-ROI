import PageLayout from '@/components/layout/PageLayout';
import ModelCard from '@/components/insights/ModelCard';
import FeatureImportanceChart from '@/components/insights/FeatureImportanceChart';
import TechnicalSpecs from '@/components/insights/TechnicalSpecs';

const MOCK_FEATURE_IMPORTANCE = [
  { feature: 'Time Saved (hours/month)', importance: 0.245 },
  { feature: 'Revenue Investment Ratio', importance: 0.118 },
  { feature: 'Time Efficiency', importance: 0.095 },
  { feature: 'Deployment Speed', importance: 0.082 },
  { feature: 'Revenue Time Interaction', importance: 0.071 },
  { feature: 'Log Investment', importance: 0.058 },
];

const MOCK_METRICS = [
  {
    label: 'Accuracy',
    value: '86.41%',
    description: 'Binary classification accuracy (High vs Not-High ROI)',
  },
  {
    label: 'AUC-ROC',
    value: '91.13%',
    description: 'Area under ROC curve - discrimination ability',
  },
  {
    label: 'Precision (High)',
    value: '82.35%',
    description: 'Accuracy when predicting High ROI projects',
  },
  {
    label: 'Training Samples',
    value: '514',
    description: 'AI deployment cases after preprocessing',
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
                  Binary Classification
                </div>
                <p className="text-sm text-[#e8dfd5] leading-relaxed font-light">
                  Predicts High ROI (≥145.5%) vs Not-High. 500 trees, depth 8, learning rate 0.03. Outperforms regression by 329%
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
              Feature importance shows time savings and early deployment signals are the strongest predictors. 
              The model achieves statistically significant performance, making it suitable for decision-support applications.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
