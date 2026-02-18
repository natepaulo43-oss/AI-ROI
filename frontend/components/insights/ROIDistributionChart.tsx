'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

const data = [
  { range: '-30 to 0%', count: 8, label: '-30-0%' },
  { range: '0 to 50%', count: 76, label: '0-50%' },
  { range: '50 to 100%', count: 118, label: '50-100%' },
  { range: '100 to 145%', count: 132, label: '100-145%' },
  { range: '145 to 200%', count: 94, label: '145-200%', highlight: true },
  { range: '200 to 300%', count: 52, label: '200-300%', highlight: true },
  { range: '300 to 500%', count: 22, label: '300-500%', highlight: true },
  { range: '500+%', count: 12, label: '500+%', highlight: true },
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#2d2520] border border-[#6b5d4f] px-4 py-2 rounded-lg">
        <p className="text-[#f5f1ed] text-sm font-light">{payload[0].payload.range}</p>
        <p className="text-[#d4a574] text-sm font-normal">{payload[0].value} projects</p>
      </div>
    );
  }
  return null;
};

export default function ROIDistributionChart() {
  return (
    <div className="grid grid-cols-12 gap-12">
      <div className="col-span-4">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-3">Figure 1</div>
        <h3 className="text-3xl font-light text-[#f5f1ed] mb-6 leading-tight">
          ROI Distribution
        </h3>
        <p className="text-sm text-[#e8dfd5] font-light leading-relaxed mb-4">
          Distribution of ROI outcomes across 514 AI deployments. The high variance (σ=215.9%, μ=122.9%) 
          illustrates why binary classification outperforms regression.
        </p>
        <div className="space-y-2 text-xs text-[#b8a894] font-light">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#d4a574]"></div>
            <span>High ROI (≥145.5%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#6b5d4f]"></div>
            <span>Not-High ROI</span>
          </div>
        </div>
      </div>

      <div className="col-span-7 col-start-6">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a3f35" />
            <XAxis 
              dataKey="label" 
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis 
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              label={{ value: 'Number of Projects', angle: -90, position: 'insideLeft', fill: '#8a7a68', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x="100-145%" stroke="#d4a574" strokeDasharray="3 3" />
            <Bar 
              dataKey="count" 
              radius={[4, 4, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.highlight ? '#d4a574' : '#6b5d4f'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-[#8a7a68] text-center mt-2 font-light">
          Threshold at 145.5% separates High ROI from Not-High ROI classifications
        </p>
      </div>
    </div>
  );
}
