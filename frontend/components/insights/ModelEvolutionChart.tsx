'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const data = [
  { 
    model: 'Regression', 
    r2: 0.13, 
    accuracy: null,
    description: 'R² = 0.13',
    color: '#6b5d4f',
    status: 'Unusable'
  },
  { 
    model: '3-Class', 
    r2: null,
    accuracy: 0.58,
    description: 'Acc = 58%',
    color: '#8a7a68',
    status: 'Marginal'
  },
  { 
    model: 'Binary', 
    r2: null,
    accuracy: 0.864,
    description: 'Acc = 86.4%',
    color: '#d4a574',
    status: 'Production'
  },
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const item = payload[0].payload;
    return (
      <div className="bg-[#2d2520] border border-[#6b5d4f] px-4 py-3 rounded-lg">
        <p className="text-[#f5f1ed] text-sm font-normal mb-1">{item.model}</p>
        <p className="text-[#e8dfd5] text-xs font-light mb-1">{item.description}</p>
        <p className="text-[#8a7a68] text-xs font-light">{item.status}</p>
      </div>
    );
  }
  return null;
};

export default function ModelEvolutionChart() {
  return (
    <div className="grid grid-cols-12 gap-12">
      <div className="col-span-4">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-3">Figure 2</div>
        <h3 className="text-3xl font-light text-[#f5f1ed] mb-6 leading-tight">
          Model Evolution
        </h3>
        <p className="text-sm text-[#e8dfd5] font-light leading-relaxed mb-4">
          Performance comparison across three modeling approaches. Binary classification 
          achieved a 329% improvement over regression (0.864 vs 0.13 R²-equivalent).
        </p>
        <div className="space-y-3 text-xs font-light">
          <div>
            <div className="text-[#6b5d4f] font-normal mb-1">Regression</div>
            <div className="text-[#b8a894]">High variance made predictions unreliable</div>
          </div>
          <div>
            <div className="text-[#8a7a68] font-normal mb-1">3-Class</div>
            <div className="text-[#b8a894]">Improved but still marginal performance</div>
          </div>
          <div>
            <div className="text-[#d4a574] font-normal mb-1">Binary</div>
            <div className="text-[#b8a894]">Production-ready accuracy achieved</div>
          </div>
        </div>
      </div>

      <div className="col-span-7 col-start-6">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a3f35" />
            <XAxis 
              dataKey="model" 
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 12 }}
            />
            <YAxis 
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              label={{ value: 'Performance Score', angle: -90, position: 'insideLeft', fill: '#8a7a68', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey={(item) => item.r2 !== null ? item.r2 : item.accuracy}
              radius={[8, 8, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-[#8a7a68] text-center mt-2 font-light">
          Binary classification provides actionable guidance vs unreliable precise predictions
        </p>
      </div>
    </div>
  );
}
