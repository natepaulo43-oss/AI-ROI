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
          <div className="space-y-6">
            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Stage 1: Regression</div>
                <div className="text-[#8a7a68] text-sm mt-2">Initial Approach</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light mb-3">
                  Initial XGBoost regression model attempted to predict exact ROI percentages. 
                  Achieved only 16% R² score, indicating the model explained just 16% of variance in outcomes.
                </p>
                <p className="text-[#b8a894] text-sm font-light">
                  Result: Unusable for production (329% worse than final approach)
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Stage 2: 3-Class</div>
                <div className="text-[#8a7a68] text-sm mt-2">Classification Pivot</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light mb-3">
                  Reframed as 3-class classification (Low/Medium/High ROI) using 33rd and 67th percentile thresholds.
                  Achieved 51.6% accuracy with advanced feature engineering.
                </p>
                <p className="text-[#b8a894] text-sm font-light">
                  Result: Marginal improvement but still below acceptable threshold
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Stage 3: Binary</div>
                <div className="text-[#8a7a68] text-sm mt-2">Breakthrough</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light mb-3">
                  Binary classification (High ROI above 145.5% vs Not-High) achieved 68.82% accuracy with 70.76% AUC-ROC.
                  Statistically significant (p less than 0.001) and production-ready.
                </p>
                <p className="text-[#b8a894] text-sm font-light">
                  Result: 33% better than 3-class, 329% better than regression
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Dataset Information */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-4">Dataset Composition</h2>
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h3 className="text-[#8a7a68] text-xs uppercase tracking-widest mb-3">Training Data</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— 462 AI deployment cases after preprocessing</li>
                <li>— 309 Not-High ROI projects (67%)</li>
                <li>— 153 High ROI projects (33%)</li>
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

        {/* Optimization Techniques */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Optimization Journey</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-6">
            Over 10 advanced techniques tested to push beyond the 68.8% accuracy ceiling:
          </p>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Techniques Tested</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-1 text-sm">
                <li>— Ultra-optimized gradient boosting (1000+ trees)</li>
                <li>— Probability calibration (sigmoid & isotonic)</li>
                <li>— Optimized decision thresholds</li>
                <li>— Weighted ensemble methods</li>
                <li>— Cross-validation ensembles</li>
                <li>— Polynomial feature expansion</li>
                <li>— Complex interaction terms</li>
              </ul>
            </div>
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Key Finding</h3>
              <p className="text-[#e8dfd5] leading-relaxed font-light text-sm mb-3">
                More complexity does not equal better performance. All advanced techniques 
                performed worse than the simple baseline, with accuracy dropping to 58-67%.
              </p>
              <p className="text-[#b8a894] text-sm font-light">
                The simple binary XGBoost classifier (500 trees, depth 8) represents the 
                optimal balance for this dataset size.
              </p>
            </div>
          </div>
        </div>

        {/* Statistical Validation */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Statistical Validation</h2>
          <div className="grid grid-cols-3 gap-8">
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Permutation Test</div>
              <div className="text-2xl font-light text-[#f5f1ed] mb-2">p = 0.0099</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Model outperformed 99 out of 100 random permutations
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Binomial Test</div>
              <div className="text-2xl font-light text-[#f5f1ed] mb-2">p = 0.000183</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Less than 0.02% chance of results occurring by random guessing
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Cross-Validation</div>
              <div className="text-2xl font-light text-[#f5f1ed] mb-2">67.77% ± 5.92%</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                95% confidence interval: 60.42% to 75.12%
              </p>
            </div>
          </div>
        </div>

        {/* Why 68.8% is the Ceiling */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-4">Performance Ceiling Explanation</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
            After extensive optimization, 68.8% represents the maximum achievable accuracy with current data due to:
          </p>
          <div className="space-y-4">
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Small Dataset</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Only 462 samples limits model complexity. Need 2,000+ samples for significant improvement.
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
