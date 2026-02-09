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
              <span className="font-normal text-[#f5f1ed]">Sample size:</span> Model trained on 1,247 SME cases, which may not capture 
              all possible firm profiles and AI adoption scenarios
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Temporal scope:</span> Data collected over a specific time period; market 
              conditions and AI technologies evolve rapidly
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Geographic coverage:</span> Dataset may be geographically limited and not 
              representative of all global markets
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Self-reported data:</span> Some metrics rely on survey responses, which may 
              contain reporting biases
            </li>
          </ul>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-4 font-normal">Generalizability Concerns</h2>
          <p className="text-[#e8e0d5] leading-relaxed font-light mb-4">
            The model's predictions are most reliable for firms that resemble the training data:
          </p>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Small to medium-sized enterprises (10-500 employees)</li>
            <li>— Industries represented in the training dataset</li>
            <li>— AI adoption patterns similar to historical cases</li>
            <li>— Investment levels within observed ranges</li>
          </ul>
          <p className="text-[#e8e0d5] leading-relaxed font-light mt-4">
            Predictions for firms outside these parameters should be treated with additional caution.
          </p>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-4 font-normal">Bias Considerations</h2>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-3">
            <li>
              <span className="font-normal text-[#f5f1ed]">Selection bias:</span> Firms that adopted AI may differ systematically from 
              those that did not
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Survivorship bias:</span> Dataset may underrepresent firms that failed after 
              AI adoption
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Industry representation:</span> Some industries may be over- or under-represented 
              in the training data
            </li>
            <li>
              <span className="font-normal text-[#f5f1ed]">Success metrics:</span> ROI is measured financially; non-financial benefits 
              and costs may not be fully captured
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
            <li>— Understand the model's limitations before making decisions</li>
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
