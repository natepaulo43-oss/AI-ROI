export default function Footer() {
  return (
    <footer className="border-t border-[#5a4d3f]/40 mt-auto">
      <div className="mx-auto max-w-6xl px-4 sm:px-8 py-12">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 sm:gap-0 text-[#b8a894]">
          <div className="text-xs uppercase tracking-widest">
            <p>AI Adoption ROI Prediction Tool</p>
            <p className="mt-2 text-[#8a7a68]">Presented at SEDSI Conference · 2026</p>
          </div>
          <div className="text-xs text-[#8a7a68]">
            Nate Paulo, Rollins AI Business Association
          </div>
        </div>
      </div>
    </footer>
  );
}
