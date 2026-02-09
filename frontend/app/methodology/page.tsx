import PageLayout from '@/components/layout/PageLayout';

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
              The model was trained on empirical data collected from small and medium-sized enterprises 
              that have implemented AI technologies. Data sources include:
            </p>
            <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
              <li>— Survey responses from 1,247 SMEs across multiple industries</li>
              <li>— Financial performance metrics pre- and post-AI adoption</li>
              <li>— Operational characteristics and maturity assessments</li>
              <li>— AI investment levels and implementation strategies</li>
            </ul>
          </div>
          <div className="col-span-6 col-start-7">
            <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-[4/3] flex items-center justify-center p-8">
              <div className="text-center">
                <div className="text-[3rem] font-light text-[#f5f1ed] mb-2">1,247</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894]">
                  SME Cases
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
              <li>— Missing value imputation using domain-appropriate methods</li>
              <li>— Outlier detection and treatment for financial metrics</li>
              <li>— Categorical encoding for industry and use case variables</li>
              <li>— Feature scaling for numerical variables</li>
              <li>— Train-test split (80/20) with stratification by industry</li>
            </ul>
          </div>
          <div className="col-span-5 col-start-8">
            <h2 className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">03 — Training</h2>
            <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
              The XGBoost regression model was trained using the following approach:
            </p>
            <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
              <li>— 5-fold cross-validation on training set</li>
              <li>— Hyperparameter tuning via grid search</li>
              <li>— Early stopping to prevent overfitting</li>
              <li>— Final evaluation on held-out test set</li>
            </ul>
          </div>
        </div>

        {/* Evaluation metrics - large display */}
        <div className="grid grid-cols-12 gap-12 items-center">
          <div className="col-span-4">
            <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-4">
              04 — Performance
            </div>
            <h2 className="text-2xl font-light text-[#f5f1ed] mb-4">
              Evaluation Metrics
            </h2>
          </div>
          <div className="col-span-7 col-start-6">
            <div className="grid grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-[2.5rem] font-light text-[#f5f1ed] tabular-nums">0.82</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">R² Score</div>
              </div>
              <div className="text-center">
                <div className="text-[2.5rem] font-light text-[#f5f1ed] tabular-nums">4.3%</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">RMSE</div>
              </div>
              <div className="text-center">
                <div className="text-[2.5rem] font-light text-[#f5f1ed] tabular-nums">3.1%</div>
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mt-2">MAE</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
