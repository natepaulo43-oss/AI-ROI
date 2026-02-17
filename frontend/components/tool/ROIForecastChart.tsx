'use client';

import { MonthlyForecast } from '@/lib/api';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine,
  Legend
} from 'recharts';

interface ROIForecastChartProps {
  forecastData: MonthlyForecast[];
  predictedROI: number;
  threshold: number;
}

export default function ROIForecastChart({ 
  forecastData, 
  predictedROI,
  threshold 
}: ROIForecastChartProps) {
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-[#2a2420] border border-[#6b5d4f]/40 rounded-lg p-3 shadow-xl">
          <p className="text-[0.65rem] uppercase tracking-[0.15em] text-[#b8a894] mb-2">
            Month {data.month}
          </p>
          <div className="space-y-1">
            <p className="text-sm text-[#f5f1ed] font-medium">
              ROI: {data.roi.toFixed(1)}%
            </p>
            <p className="text-xs text-[#e8dfd5]/70">
              Range: {data.lower.toFixed(1)}% - {data.upper.toFixed(1)}%
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomLegend = () => (
    <div className="flex items-center justify-center gap-6 mt-4">
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-[#d4a574]"></div>
        <span className="text-[0.65rem] uppercase tracking-[0.15em] text-[#b8a894]">
          Predicted ROI
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-[#8a7a68]/40"></div>
        <span className="text-[0.65rem] uppercase tracking-[0.15em] text-[#b8a894]">
          Confidence Interval
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-8 h-0.5 bg-[#6b5d4f] border-t-2 border-dashed border-[#b8a894]"></div>
        <span className="text-[0.65rem] uppercase tracking-[0.15em] text-[#b8a894]">
          High ROI Threshold (145.5%)
        </span>
      </div>
    </div>
  );

  return (
    <div className="w-full">
      <div className="mb-6">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#b8a894] mb-2">
          12-Month ROI Forecast
        </div>
        <p className="text-xs text-[#e8dfd5]/70 leading-relaxed">
          Projected ROI trajectory with {' '}
          <span className="text-[#d4a574]">±62.67% confidence interval</span>
          {' '}(MAE). ROI typically ramps up over 6 months as the AI system matures.
        </p>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <AreaChart
          data={forecastData}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorROI" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#d4a574" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#d4a574" stopOpacity={0.05}/>
            </linearGradient>
            <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8a7a68" stopOpacity={0.15}/>
              <stop offset="95%" stopColor="#8a7a68" stopOpacity={0.02}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke="#6b5d4f" 
            opacity={0.2}
            vertical={false}
          />
          
          <XAxis 
            dataKey="month" 
            stroke="#b8a894"
            tick={{ fill: '#b8a894', fontSize: 11 }}
            tickLine={{ stroke: '#6b5d4f' }}
            axisLine={{ stroke: '#6b5d4f' }}
            label={{ 
              value: 'Month', 
              position: 'insideBottom', 
              offset: -5,
              style: { 
                fill: '#b8a894', 
                fontSize: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.15em'
              }
            }}
          />
          
          <YAxis 
            stroke="#b8a894"
            tick={{ fill: '#b8a894', fontSize: 11 }}
            tickLine={{ stroke: '#6b5d4f' }}
            axisLine={{ stroke: '#6b5d4f' }}
            label={{ 
              value: 'ROI (%)', 
              angle: -90, 
              position: 'insideLeft',
              style: { 
                fill: '#b8a894', 
                fontSize: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.15em'
              }
            }}
            tickFormatter={(value) => `${value}%`}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          <ReferenceLine 
            y={threshold} 
            stroke="#b8a894" 
            strokeDasharray="5 5"
            strokeWidth={1.5}
            opacity={0.6}
          />
          
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#colorConfidence)"
            fillOpacity={1}
          />
          
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#3d342a"
            fillOpacity={1}
          />
          
          <Area
            type="monotone"
            dataKey="roi"
            stroke="#d4a574"
            strokeWidth={2.5}
            fill="url(#colorROI)"
            fillOpacity={1}
          />
        </AreaChart>
      </ResponsiveContainer>

      <CustomLegend />

      {forecastData && forecastData.length > 0 && (
        <div className="mt-6 pt-6 border-t border-[#6b5d4f]/40">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-[0.6rem] uppercase tracking-[0.15em] text-[#b8a894] mb-1">
                Stabilized ROI
              </div>
              <div className="text-lg font-light text-[#f5f1ed]">
                {predictedROI.toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-[0.6rem] uppercase tracking-[0.15em] text-[#b8a894] mb-1">
                Month 1 ROI
              </div>
              <div className="text-lg font-light text-[#f5f1ed]">
                {forecastData[0]?.roi?.toFixed(1) || '0.0'}%
              </div>
            </div>
            <div>
              <div className="text-[0.6rem] uppercase tracking-[0.15em] text-[#b8a894] mb-1">
                Month 12 ROI
              </div>
              <div className="text-lg font-light text-[#f5f1ed]">
                {forecastData[11]?.roi?.toFixed(1) || '0.0'}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
