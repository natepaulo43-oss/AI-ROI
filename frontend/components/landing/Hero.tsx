export default function Hero() {
  return (
    <div className="relative min-h-[60vh] sm:min-h-[85vh] flex items-center">
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-8 w-full items-center">
        {/* Left column: Dominant typography */}
        <div className="col-span-1 sm:col-span-5">
          <div className="mb-6 text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68]">
            Research Tool
          </div>

          <h1 className="text-[3rem] sm:text-[5.5rem] font-light text-[#f5f1ed] leading-[0.95] tracking-tight mb-8 sm:mb-12">
            AI Adoption:
            <br />
            Predicting ROI
          </h1>

          <div className="space-y-1 text-[0.7rem] uppercase tracking-[0.15em] text-[#b8a894] mb-8">
            <div>Binary Classification Model</div>
            <div className="flex items-center gap-3">
              <span>76.7% Accuracy</span>
              <span className="text-[#8a7a68]">—</span>
              <span>75.5% Avg Confidence</span>
            </div>
          </div>
        </div>

        {/* Right column: Visual element */}
        <div className="col-span-1 sm:col-span-6 sm:col-start-7">
          <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-[4/3] flex items-center justify-center">
            <div className="text-center px-8 sm:px-12">
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

      {/* Vertical metadata — hidden on mobile to prevent horizontal overflow */}
      <div className="hidden sm:block absolute left-0 top-1/2 -translate-y-1/2 -translate-x-8">
        <div className="rotate-180 text-[0.6rem] uppercase tracking-[0.2em] text-[#8a7a68]" style={{ writingMode: 'vertical-rl' }}>
          2026 Research Project
        </div>
      </div>
    </div>
  );
}
