import PageLayout from '@/components/layout/PageLayout';

export default function Limitations() {
  return (
    <div className="mx-auto max-w-7xl px-12 py-16">
      {/* Page header */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          Responsible AI
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          Limitations
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Understanding the boundaries and responsible use of this prediction tool
        </p>
      </div>

      {/* Asymmetric content sections */}
      <div className="space-y-24">
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-3">Important Notice</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light">
            This tool is designed as a decision-support system, not a definitive 
            predictor of outcomes. All predictions should be interpreted with appropriate caution 
            and validated against domain expertise and organizational context.
          </p>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-4 font-normal">Data Limitations</h2>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-3">
            <li>
              <span className="font-normal text-[#f5f1ed]">Limited sample size:</span> Training data size limits 
              model complexity and may not capture all possible AI adoption scenarios
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Moderate accuracy:</span> Predictions have a meaningful error rate and 
              should be used as decision support, not as the sole decision-maker
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Missing critical features:</span> Team expertise, data quality, organizational 
              readiness, and execution factors not captured in dataset
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">High ROI variance:</span> ROI ranges from -30% to 3,750% with high inherent 
              unpredictability due to execution and external factors
            </li>
          </ul>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-4 font-normal">Model Performance Ceiling</h2>
          <p className="text-[#e8e0d5] leading-relaxed font-light mb-4">
            Current performance represents the ceiling achievable with available data. The model shows:
          </p>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Higher accuracy on Not-High projects (85.5%) than High ROI projects (61.3%)</li>
            <li>— Statistically significant but moderate effect size</li>
            <li>— More complexity does not improve performance (overfitting risk)</li>
          </ul>
          <p className="text-[#e8e0d5] leading-relaxed font-light mt-4">
            Further improvement requires more data (2,000+ samples) and critical missing features (team quality, execution factors). 
            See the About and Methodology pages for detailed performance comparisons.
          </p>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-4 font-normal">Prediction Uncertainty</h2>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-3">
            <li>
              <span className="font-normal text-[#f5f1ed]">Conservative predictions:</span> Model has low false positive rate (11 cases) 
              but misses 36.11% of High ROI projects (13 false negatives)
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Precision trade-off:</span> High ROI predictions require validation for high-stakes decisions
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Cross-validation variance:</span> Performance varies across data splits; 
              95% confidence interval: 60.42% to 75.12%
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Probability scores essential:</span> Binary predictions alone insufficient; 
              must show probability (0-100%) for informed decision-making
            </li>
          </ul>
        </div>

        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-3">Human-in-the-Loop Decision Making</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-4">
            This tool is designed to augment, not replace, human judgment. Effective use requires:
          </p>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Domain expertise to interpret predictions in organizational context</li>
            <li>— Critical evaluation of model assumptions and limitations</li>
            <li>— Consideration of factors not captured by the model</li>
            <li>— Validation against multiple information sources</li>
            <li>— Ongoing monitoring of actual outcomes versus predictions</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-4 font-normal">Responsible AI Use</h2>
          <p className="text-[#e8e0d5] leading-relaxed font-light mb-4">
            Users of this tool should:
          </p>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Understand the model&apos;s limitations before making decisions</li>
            <li>— Avoid over-reliance on automated predictions</li>
            <li>— Consider ethical implications of AI adoption in their context</li>
            <li>— Maintain transparency when using model outputs to inform decisions</li>
            <li>— Regularly reassess predictions against actual outcomes</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
