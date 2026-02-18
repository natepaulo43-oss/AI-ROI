import ROIDistributionChart from '@/components/insights/ROIDistributionChart';

export default function Hypothesis() {
  return (
    <div className="mx-auto max-w-7xl px-12 py-16">
      {/* Page header */}
      <div className="mb-20">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-6">
          Why This Matters
        </div>
        <h1 className="text-[4rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-4">
          Research Motivations
        </h1>
        <p className="text-sm text-[#e8dfd5] font-light max-w-md">
          Understanding the gap in AI ROI guidance for small and medium enterprises
        </p>
      </div>

      {/* Content sections */}
      <div className="space-y-24">
        {/* The Problem */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">The Problem</h2>
          <div className="space-y-6">
            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-3xl font-light text-[#f5f1ed]">~99%</div>
                <div className="text-[#8a7a68] text-sm mt-2">of U.S. Businesses</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light">
                  Small and medium enterprises (SMEs) make up the vast majority of U.S. businesses, 
                  yet they face unique challenges when evaluating AI investments that larger corporations do not.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Resource Gap</div>
                <div className="text-[#8a7a68] text-sm mt-2">Consulting Divide</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light mb-3">
                  Large corporations can hire expensive consulting firms to measure and predict AI ROI. 
                  These firms provide detailed analysis, risk assessments, and implementation roadmaps.
                </p>
                <p className="text-[#b8a894] text-sm font-light">
                  SMEs usually cannot afford that level of guidance, leaving them to make critical 
                  investment decisions without data-driven support.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Uncertainty</div>
                <div className="text-[#8a7a68] text-sm mt-2">Adoption Barrier</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light">
                  ROI uncertainty is one of the biggest barriers to AI adoption among SMEs. Without 
                  clear expectations of returns, decision-makers struggle to justify the investment 
                  to stakeholders and allocate limited resources effectively.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              <div className="col-span-3">
                <div className="text-2xl font-light text-[#f5f1ed]">Data Scarcity</div>
                <div className="text-[#8a7a68] text-sm mt-2">SME-Specific Gap</div>
              </div>
              <div className="col-span-8">
                <p className="text-[#e8dfd5] leading-relaxed font-light">
                  There is little SME-specific data on measurable AI outcomes. Most published case 
                  studies and research focus on large enterprise deployments, which operate under 
                  different constraints, budgets, and organizational structures.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ROI Distribution Visualization */}
        <ROIDistributionChart />

        {/* The Goal */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-4">Research Goal</h2>
          <p className="text-[1.5rem] font-light text-[#f5f1ed] leading-relaxed mb-6">
            Build a <span className="text-[#d4a574]">data-driven AI ROI prediction tool</span> that SMEs can access — 
            without needing expensive consultants.
          </p>
          <p className="text-[#e8dfd5] leading-relaxed font-light">
            This tool democratizes access to AI investment insights, providing SME decision-makers 
            with the same level of analytical support that was previously available only to large 
            corporations with substantial consulting budgets.
          </p>
        </div>

        {/* What Makes This Different */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">What Makes This Different</h2>
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">SME-Focused Dataset</h3>
              <p className="text-[#e8dfd5] leading-relaxed font-light text-sm">
                Built on real AI deployment cases specifically from small and medium enterprises, 
                not extrapolated from large enterprise data. Covers 16 industry sectors, 15 distinct AI use cases, 
                with revenue ranging from $1.1M to $540M and investments from $10.8K to $2.16M.
              </p>
            </div>
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Practical Approach</h3>
              <p className="text-[#e8dfd5] leading-relaxed font-light text-sm mb-3">
                Binary classification (High ROI vs Not-High) provides actionable guidance rather 
                than unreliable precise predictions.
              </p>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Statistically significant performance</li>
                <li>— Clear decision support framework</li>
                <li>— Transparent limitations communicated</li>
                <li>— Free and accessible to all SMEs</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Research Hypothesis */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Research Hypothesis</h2>
          <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
            <div className="mb-6">
              <div className="text-[#8a7a68] text-xs uppercase tracking-widest mb-3">Alternative Hypothesis (H₁)</div>
              <p className="text-lg font-light text-[#f5f1ed] leading-relaxed">
                SMEs that implement AI with clear operational alignment, structured deployment, and defined use cases 
                are more likely to achieve high ROI than SMEs that adopt AI without a defined strategy.
              </p>
            </div>
            <div>
              <div className="text-[#8a7a68] text-xs uppercase tracking-widest mb-3">Null Hypothesis (H₀)</div>
              <p className="text-lg font-light text-[#f5f1ed] leading-relaxed">
                There is no statistically significant relationship between the structure and strategic alignment of AI 
                implementation and ROI outcomes in SMEs.
              </p>
            </div>
          </div>
        </div>

        {/* Impact & Applications */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Impact & Applications</h2>
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">For SME Decision-Makers</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Evaluate AI investments with data-driven confidence</li>
                <li>— Prioritize projects by predicted ROI probability</li>
                <li>— Set realistic stakeholder expectations</li>
                <li>— Identify key success factors before deployment</li>
                <li>— Access enterprise-level insights without consulting fees</li>
              </ul>
            </div>
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">For the Research Community</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Demonstrates binary classification superiority for ROI prediction</li>
                <li>— Provides SME-specific AI adoption dataset</li>
                <li>— Identifies execution quality as primary success factor</li>
                <li>— Establishes performance baseline for future work</li>
                <li>— Highlights critical missing features for improvement</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Key Insights from Research */}
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h2 className="text-sm font-normal text-[#f5f1ed] mb-4">Key Insights from Research</h2>
          <div className="space-y-4">
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Execution Matters More Than Planning</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Early operational signals (time savings, revenue increases) are 3.4x more predictive 
                than pre-adoption characteristics like investment size or company revenue.
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Investment Efficiency Over Absolute Investment</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Investment as a percentage of revenue is more predictive than total investment amount. 
                SMEs should focus on right-sizing investments relative to their scale.
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">Speed Correlates with Success</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                Faster deployment times correlate with higher ROI. Extended planning phases do not 
                guarantee better outcomes and may indicate organizational friction.
              </p>
            </div>
            <div>
              <div className="text-[#f5f1ed] text-sm mb-2 font-normal">High Variance is Inherent</div>
              <p className="text-[#e8dfd5] text-sm font-light">
                ROI outcomes vary wildly even within the same sector and company size. This reflects 
                the importance of factors not captured in typical pre-adoption data.
              </p>
            </div>
          </div>
        </div>

        {/* Future Directions */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Future Research Directions</h2>
          <p className="text-[#e8dfd5] leading-relaxed font-light mb-6">
            To improve prediction accuracy, future research should focus on:
          </p>
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Data Collection</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Expand dataset to 2,000+ samples</li>
                <li>— Capture team expertise and experience levels</li>
                <li>— Measure data quality and availability</li>
                <li>— Track organizational readiness scores</li>
                <li>— Document change management effectiveness</li>
              </ul>
            </div>
            <div>
              <h3 className="text-[#f5f1ed] text-sm mb-3 font-normal">Methodological Advances</h3>
              <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2 text-sm">
                <li>— Develop hybrid binary + regression approach</li>
                <li>— Implement probability calibration techniques</li>
                <li>— Create sector-specific prediction models</li>
                <li>— Add confidence intervals to predictions</li>
                <li>— Explore causal inference methods</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
