import DirectionalIndicator from './DirectionalIndicator';

interface PredictionCardProps {
  predictedRoi: number;
  direction: 'positive' | 'neutral' | 'negative';
}

export default function PredictionCard({ predictedRoi, direction }: PredictionCardProps) {
  return (
    <div className="py-12 border-t border-b border-stone-200">
      <p className="result-label mb-4">Predicted ROI</p>
      <p className="result-metric mb-6">{predictedRoi.toFixed(1)}%</p>
      <DirectionalIndicator direction={direction} />
    </div>
  );
}
