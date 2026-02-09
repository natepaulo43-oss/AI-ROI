export default function Hero() {
  return (
    <div className="relative min-h-[85vh] flex items-center">
      {/* Asymmetric grid: large title left, visual space right */}
      <div className="grid grid-cols-12 gap-8 w-full items-center">
        {/* Left column: Dominant typography (5 columns) */}
        <div className="col-span-5">
          <div className="mb-6 text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68]">
            Research Tool
          </div>
          
          <h1 className="text-[5.5rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-12">
            AI Adoption
            <br />
            ROI
            <br />
            Prediction
          </h1>
          
          <div className="space-y-1 text-[0.7rem] uppercase tracking-[0.15em] text-[#b8a894] mb-8">
            <div>SME Decision Support</div>
            <div className="flex items-center gap-3">
              <span>Machine Learning</span>
              <span className="text-[#8a7a68]">—</span>
              <span>XGBoost</span>
            </div>
          </div>
        </div>

        {/* Right column: Visual element (6 columns) */}
        <div className="col-span-6 col-start-7">
          <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-[4/3] flex items-center justify-center">
            <div className="text-center px-12">
              <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-4">
                Presented at SEDSI Conference
              </div>
              <p className="text-sm text-[#e8dfd5] leading-relaxed font-light">
                A data-driven decision-support system for predicting return on investment 
                from AI adoption in small and medium enterprises
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Vertical metadata on far left edge */}
      <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-8">
        <div className="rotate-180 text-[0.6rem] uppercase tracking-[0.2em] text-[#8a7a68]" style={{ writingMode: 'vertical-rl' }}>
          2026 Research Project
        </div>
      </div>
    </div>
  );
}
