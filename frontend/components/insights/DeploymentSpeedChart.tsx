'use client';

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ZAxis } from 'recharts';

const scatterData = [
  { days: 30, roi: 285, size: 80, category: 'high' },
  { days: 45, roi: 320, size: 60, category: 'high' },
  { days: 60, roi: 195, size: 90, category: 'high' },
  { days: 75, roi: 240, size: 70, category: 'high' },
  { days: 90, roi: 175, size: 85, category: 'high' },
  { days: 105, roi: 160, size: 65, category: 'high' },
  { days: 120, roi: 155, size: 75, category: 'high' },
  { days: 150, roi: 150, size: 55, category: 'high' },
  
  { days: 40, roi: 125, size: 60, category: 'not-high' },
  { days: 55, roi: 110, size: 70, category: 'not-high' },
  { days: 70, roi: 95, size: 50, category: 'not-high' },
  { days: 85, roi: 85, size: 65, category: 'not-high' },
  { days: 100, roi: 75, size: 55, category: 'not-high' },
  { days: 130, roi: 60, size: 75, category: 'not-high' },
  { days: 160, roi: 45, size: 60, category: 'not-high' },
  { days: 180, roi: 35, size: 50, category: 'not-high' },
  { days: 200, roi: 25, size: 65, category: 'not-high' },
  
  { days: 50, roi: 140, size: 70, category: 'not-high' },
  { days: 95, roi: 135, size: 60, category: 'not-high' },
  { days: 140, roi: 120, size: 55, category: 'not-high' },
  { days: 35, roi: 165, size: 80, category: 'high' },
  { days: 110, roi: 180, size: 65, category: 'high' },
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const item = payload[0].payload;
    return (
      <div className="bg-[#2d2520] border border-[#6b5d4f] px-4 py-3 rounded-lg">
        <p className="text-[#e8dfd5] text-xs font-light">Deployment: {item.days} days</p>
        <p className="text-[#d4a574] text-xs font-light">ROI: {item.roi}%</p>
        <p className="text-[#8a7a68] text-xs font-light capitalize">{item.category} ROI</p>
      </div>
    );
  }
  return null;
};

export default function DeploymentSpeedChart() {
  const highROI = scatterData.filter(d => d.category === 'high');
  const notHighROI = scatterData.filter(d => d.category === 'not-high');

  return (
    <div className="grid grid-cols-12 gap-12">
      <div className="col-span-4">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-3">Figure 7</div>
        <h3 className="text-3xl font-light text-[#f5f1ed] mb-6 leading-tight">
          Speed vs Success
        </h3>
        <p className="text-sm text-[#e8dfd5] font-light leading-relaxed mb-4">
          Deployment speed correlates with ROI outcomes. Faster deployments (30-90 days) 
          show higher success rates than extended planning phases (120+ days).
        </p>
        <div className="space-y-2 text-xs text-[#b8a894] font-light">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#d4a574]"></div>
            <span>High ROI (≥145.5%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#6b5d4f]"></div>
            <span>Not-High ROI</span>
          </div>
        </div>
        <p className="text-xs text-[#b8a894] font-light mt-4">
          Extended planning does not guarantee better outcomes and may indicate organizational friction.
        </p>
      </div>

      <div className="col-span-7 col-start-6">
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a3f35" />
            <XAxis 
              type="number"
              dataKey="days"
              domain={[0, 220]}
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              label={{ value: 'Days to Deployment', position: 'insideBottom', offset: -10, fill: '#8a7a68', fontSize: 11 }}
            />
            <YAxis 
              type="number"
              dataKey="roi"
              domain={[0, 350]}
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              label={{ value: 'ROI (%)', angle: -90, position: 'insideLeft', fill: '#8a7a68', fontSize: 11 }}
            />
            <ZAxis range={[40, 100]} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={145.5} stroke="#d4a574" strokeDasharray="3 3" />
            <Scatter 
              data={notHighROI} 
              fill="#6b5d4f"
              fillOpacity={0.6}
            />
            <Scatter 
              data={highROI} 
              fill="#d4a574"
              fillOpacity={0.8}
            />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-[#8a7a68] text-center mt-2 font-light">
          Dashed line at 145.5% represents the High ROI threshold
        </p>
      </div>
    </div>
  );
}
