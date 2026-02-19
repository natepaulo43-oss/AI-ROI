import PageLayout from '@/components/layout/PageLayout';
import ModelEvolutionChart from '@/components/insights/ModelEvolutionChart';
import LearningCurveChart from '@/components/insights/LearningCurveChart';

export default function Methodology() {
  return (
    <div className="mx-auto max-w-7xl px-12 py-16">
      {/* Page header */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          Research Methods
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          Methodology
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Data pipeline and training approach for the AI ROI prediction model
        </p>
      </div>

      {/* Asymmetric content sections */}
      <div className="space-y-24">
        {/* Data sources with visual */}
        <div className="grid grid-cols-12 gap-12">
          <div className="col-span-5">
            <h2 className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">01 — Data Sources</h2>
            <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
              The model was trained on real-world AI deployment data including Fortune 500 case studies 
              and synthetic data based on industry research. Data sources include:
            </p>
            <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
              <li>— 514 AI deployment cases after preprocessing</li>
              <li>— Fortune 500 implementations (Klarna, Alibaba, JPMorgan, Walmart)</li>
              <li>— 200+ cases from McKinsey, Gartner, BCG research</li>
              <li>— 16 industry sectors and 15 distinct AI use cases</li>
              <li>— ROI range: -30% to 3,750% (high variance)</li>
            </ul>
          </div>
          <div className="col-span-6 col-start-7">
            <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-[4/3] flex items-center justify-center p-8">
              <div className="text-center">
                <div className="text-[3rem] font-light text-[#f5f1ed] mb-2">514</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894]">
                  Training Cases
                </div>
                <div className="text-sm text-[#e8dfd5] mt-4 font-light">
                  335 Not-High (65%) | 179 High (35%)
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Preprocessing */}
        <div className="grid grid-cols-12 gap-12">
          <div className="col-span-6">
            <h2 className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">02 — Preprocessing</h2>
            <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
              Raw data underwent rigorous preprocessing to ensure model quality:
            </p>
            <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
              <li>— Outlier removal (ROI outside -100% to 500% range)</li>
              <li>— Binary classification threshold at 145.5% ROI (65th percentile)</li>
              <li>— Feature engineering: 18 base features to 57 after encoding</li>
              <li>— One-hot encoding for 5 categorical variables</li>
              <li>— StandardScaler normalization for numeric features</li>
              <li>— Stratified sampling by ROI class for balanced validation</li>
            </ul>
          </div>
          <div className="col-span-5 col-start-8">
            <h2 className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">03 — Model Selection</h2>
            <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
              Binary classification was chosen after extensive testing:
            </p>
            <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
              <li>— Regression: 16% R² (unusable)</li>
              <li>— 3-class classification: 51.6% accuracy</li>
              <li>— Binary classification: 76.7% accuracy (winner)</li>
              <li>— Gradient Boosting: 500 trees, depth 5, learning rate 0.05</li>
              <li>— Class balancing: scale_pos_weight=1.87</li>
              <li>— 5-fold stratified cross-validation</li>
            </ul>
          </div>
        </div>

        {/* Model Evolution Visualization */}
        <ModelEvolutionChart />

        {/* Learning Curve Visualization */}
        <LearningCurveChart />

        {/* Evaluation metrics - large display */}
        <div className="grid grid-cols-12 gap-12 items-center">
          <div className="col-span-4">
            <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">
              04 — Performance
            </div>
            <h2 className="text-2xl font-light text-[#f5f1ed] mb-4">
              Binary Classification Results
            </h2>
            <p className="text-[#b8a894] text-xs font-light mb-2">
              Gradient Boosting Algorithm
            </p>
            <p className="text-[#e8dfd5] text-sm font-light">
              Validated with 5-fold cross-validation
            </p>
          </div>
          <div className="col-span-7 col-start-6">
            <div className="grid grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-[2.5rem] font-light text-[#f5f1ed] tabular-nums">76.70%</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-[2.5rem] font-light text-[#f5f1ed] tabular-nums">76.74%</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">AUC-ROC</div>
              </div>
              <div className="text-center">
                <div className="text-[2.5rem] font-light text-[#f5f1ed] tabular-nums">75.50%</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">Avg Confidence</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-8 mt-8">
              <div className="text-center">
                <div className="text-xl font-light text-[#f5f1ed] tabular-nums">63.89%</div>
                <div className="text-[0.6rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">Recall (High ROI)</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-light text-[#f5f1ed] tabular-nums">68.12% ± 2.71%</div>
                <div className="text-[0.6rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">Cross-Validation</div>
              </div>
            </div>
          </div>
        </div>

        {/* Additional context */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-3">Model Comparison</h2>
          <div className="grid grid-cols-3 gap-8">
            <div>
              <div className="text-[#8a7a68] text-xs uppercase tracking-widest mb-2">Regression</div>
              <div className="text-2xl font-light text-[#f5f1ed] mb-1">16% R²</div>
              <p className="text-[#e8dfd5] text-sm font-light">Unusable for production</p>
            </div>
            <div>
              <div className="text-[#8a7a68] text-xs uppercase tracking-widest mb-2">3-Class</div>
              <div className="text-2xl font-light text-[#f5f1ed] mb-1">51.6%</div>
              <p className="text-[#e8dfd5] text-sm font-light">Marginal improvement</p>
            </div>
            <div>
              <div className="text-[#8a7a68] text-xs uppercase tracking-widest mb-2">Binary (Final)</div>
              <div className="text-2xl font-light text-[#f5f1ed] mb-1">76.7%</div>
              <div className="text-[#e8dfd5] text-sm font-light">Production-ready</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
