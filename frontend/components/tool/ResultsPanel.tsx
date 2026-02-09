import DirectionalIndicator from './DirectionalIndicator';

interface ResultsPanelProps {
  predictedRoi: number;
  direction: 'positive' | 'neutral' | 'negative';
  interpretation: string;
}

export default function ResultsPanel({ predictedRoi, direction, interpretation }: ResultsPanelProps) {
  return (
    <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] p-12 min-h-[600px] flex flex-col justify-between">
      {/* Massive ROI display */}
      <div>
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-6">
          Predicted Return
        </div>
        <div className="text-[6rem] font-light text-[#f5f1ed] leading-[0.9] tracking-tight mb-6">
          {predictedRoi.toFixed(1)}%
        </div>
        <DirectionalIndicator direction={direction} />
      </div>

      {/* Interpretation at bottom */}
      <div className="mt-12 pt-8 border-t border-[#6b5d4f]/40">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-3">
          Interpretation
        </div>
        <p className="text-sm text-[#e8dfd5] leading-relaxed font-light">
          {interpretation}
        </p>
      </div>
    </div>
  );
}
