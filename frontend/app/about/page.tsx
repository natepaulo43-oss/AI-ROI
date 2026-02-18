export default function About() {
  return (
    <div className="mx-auto max-w-7xl px-12 py-16">
      {/* Page header */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          Research Methodology
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          About This Model
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Understanding the development process, methodology, and evolution of the AI ROI prediction system
        </p>
      </div>

      {/* Content sections */}
      <div className="space-y-24">
        {/* Model Evolution */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Model Evolution</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-6">
            The model went through three major iterations before reaching production readiness:
          </p>
          <div className="space-y-6">
            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Regression</div>
                <div className="text-[#8a7a68] text-sm mt-2">Initial Approach</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light">
                  Attempted to predict exact ROI percentages but proved unusable due to high variance in outcomes.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">3-Class</div>
                <div className="text-[#8a7a68] text-sm mt-2">Classification Pivot</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light">
                  Reframed as Low/Medium/High ROI classification but still showed marginal performance.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Binary</div>
                <div className="text-[#8a7a68] text-sm mt-2">Final Solution</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light">
                  Binary classification (High ROI ≥145.5% vs Not-High) achieved production-ready performance and statistical significance.
                </p>
              </div>
            </div>
          </div>
          <p className="text-[#b8a894] text-sm font-light mt-6">
            See the Methodology page for detailed performance metrics and technical specifications.
          </p>
        </div>

        {/* Dataset Information */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-4">Dataset Composition</h2>
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h3 className="text-[#8a7a68] text-xs uppercase tracking-widest mb-3">Training Data</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— 514 AI deployment cases after preprocessing</li>
                <li>— 335 Not-High ROI projects (65%)</li>
                <li>— 179 High ROI projects (35%)</li>
                <li>— 16 industry sectors represented</li>
                <li>— 15 distinct AI use cases</li>
              </ul>
            </div>
            <div>
              <h3 className="text-[#8a7a68] text-xs uppercase tracking-widest mb-3">Notable Cases</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Klarna: 40M profit improvement</li>
                <li>— Alibaba: 150M annual savings</li>
                <li>— JPMorgan: 1.5B saved, 20% revenue boost</li>
                <li>— Walmart: 25% supply chain cost reduction</li>
                <li>— Plus 200+ synthetic cases from research</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Optimization Journey */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Optimization Journey</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
            Over 10 advanced techniques were tested to improve performance, including ultra-optimized gradient boosting, 
            probability calibration, ensemble methods, and polynomial feature expansion.
          </p>
          <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
            <h3 className="text-[#f5f1ed] text-sm mb-2 font-normal">Key Finding</h3>
            <p className="text-[#e8dfd5] leading-relaxed font-light text-sm">
              More complexity does not equal better performance. All advanced techniques performed worse than the simple baseline. 
              The simple binary XGBoost classifier represents the optimal balance for this dataset size.
            </p>
          </div>
        </div>


        {/* Model Performance */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-4">Current Performance</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
            The binary XGBoost classifier achieves 86.4% accuracy with 91.1% AUC-ROC on the current dataset:
          </p>
          <div className="space-y-4">
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Strong Predictive Power</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                514 samples with balanced features enable reliable High vs Not-High ROI classification.
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">High Inherent Variance</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                ROI ranges from -30% to 3,750% with standard deviation (215.9%) exceeding mean (122.9%).
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Missing Critical Features</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Team expertise, data quality, organizational readiness, and execution factors not captured in dataset.
              </p>
            </div>
          </div>
        </div>

        {/* Conference Context */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Research Contribution</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
            This research demonstrates that binary classification significantly outperforms regression 
            for AI ROI prediction, achieving production-ready accuracy despite inherent data limitations. 
            The model provides actionable decision support for SMEs evaluating AI investments.
          </p>
          <div className="grid grid-cols-2 gap-8 mt-8">
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Practical Applications</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Portfolio prioritization and ranking</li>
                <li>— Risk assessment for AI projects</li>
                <li>— Resource allocation optimization</li>
                <li>— Expectation setting with stakeholders</li>
              </ul>
            </div>
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Future Directions</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Expand dataset to 2,000+ samples</li>
                <li>— Add team and execution quality features</li>
                <li>— Develop hybrid binary + regression approach</li>
                <li>— Implement probability calibration</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
