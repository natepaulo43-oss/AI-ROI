import DirectionalIndicator from './DirectionalIndicator';
import ROIForecastChart from './ROIForecastChart';
import { MonthlyForecast } from '@/lib/api';

interface ResultsPanelProps {
  predictedRoi: number;
  direction: 'high' | 'not-high';
  interpretation: string;
  forecastData: MonthlyForecast[];
  threshold: number;
  confidence: number;
}

export default function ResultsPanel({ 
  predictedRoi, 
  direction, 
  interpretation,
  forecastData,
  threshold,
  confidence
}: ResultsPanelProps) {
  return (
    <div className="space-y-6">
      {/* Top card: ROI display */}
      <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] p-10">
        <div className="grid grid-cols-2 gap-8">
          {/* Left: Massive ROI display */}
          <div>
            <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-4">
              Predicted Return
            </div>
            <div className="text-[5rem] font-light text-[#f5f1ed] leading-[0.9] tracking-tight mb-4">
              {predictedRoi.toFixed(1)}%
            </div>
            <DirectionalIndicator direction={direction} />
          </div>

          {/* Right: Key metrics */}
          <div className="flex flex-col justify-center space-y-4">
            <div>
              <div className="text-[0.6rem] uppercase tracking-[0.15em] text-[#b8a894] mb-1">
                Classification
              </div>
              <div className="text-2xl font-light text-[#f5f1ed]">
                {predictedRoi >= threshold ? 'High ROI' : 'Not-High ROI'}
              </div>
            </div>
          </div>
        </div>

        {/* Interpretation */}
        <div className="mt-8 pt-6 border-t border-[#6b5d4f]/40">
          <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-3">
            Interpretation
          </div>
          <p className="text-sm text-[#e8dfd5] leading-relaxed font-light">
            {interpretation}
          </p>
        </div>
      </div>

      {/* Bottom card: Forecast chart */}
      <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] p-10">
        <ROIForecastChart 
          forecastData={forecastData}
          predictedROI={predictedRoi}
          threshold={threshold}
        />
      </div>
    </div>
  );
}
