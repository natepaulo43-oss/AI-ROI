'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const rocData = [
  { fpr: 0.00, tpr: 0.00, randomTpr: 0.00 },
  { fpr: 0.02, tpr: 0.15, randomTpr: 0.02 },
  { fpr: 0.04, tpr: 0.32, randomTpr: 0.04 },
  { fpr: 0.06, tpr: 0.48, randomTpr: 0.06 },
  { fpr: 0.08, tpr: 0.61, randomTpr: 0.08 },
  { fpr: 0.10, tpr: 0.71, randomTpr: 0.10 },
  { fpr: 0.15, tpr: 0.82, randomTpr: 0.15 },
  { fpr: 0.20, tpr: 0.88, randomTpr: 0.20 },
  { fpr: 0.25, tpr: 0.92, randomTpr: 0.25 },
  { fpr: 0.30, tpr: 0.95, randomTpr: 0.30 },
  { fpr: 0.40, tpr: 0.97, randomTpr: 0.40 },
  { fpr: 0.50, tpr: 0.98, randomTpr: 0.50 },
  { fpr: 0.70, tpr: 0.99, randomTpr: 0.70 },
  { fpr: 1.00, tpr: 1.00, randomTpr: 1.00 },
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#2d2520] border border-[#6b5d4f] px-4 py-2 rounded-lg">
        <p className="text-[#e8dfd5] text-xs font-light">FPR: {(payload[0].payload.fpr * 100).toFixed(1)}%</p>
        <p className="text-[#d4a574] text-xs font-light">TPR: {(payload[0].payload.tpr * 100).toFixed(1)}%</p>
      </div>
    );
  }
  return null;
};

export default function ROCCurveChart() {
  return (
    <div className="grid grid-cols-12 gap-12">
      <div className="col-span-4">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-3">Figure 5</div>
        <h3 className="text-3xl font-light text-[#f5f1ed] mb-6 leading-tight">
          ROC Curve
        </h3>
        <p className="text-sm text-[#e8dfd5] font-light leading-relaxed mb-4">
          Receiver Operating Characteristic curve showing the model&apos;s discrimination ability. 
          AUC-ROC of 76.74% indicates good separation between High and Not-High ROI projects.
        </p>
        <div className="space-y-2 text-xs text-[#b8a894] font-light">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-[#d4a574]"></div>
            <span>Model (AUC = 0.7674)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-[#6b5d4f]"></div>
            <span>Random Classifier (AUC = 0.5)</span>
          </div>
        </div>
      </div>

      <div className="col-span-7 col-start-6">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={rocData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a3f35" />
            <XAxis 
              type="number"
              dataKey="fpr"
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -10, fill: '#8a7a68', fontSize: 11 }}
            />
            <YAxis 
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              stroke="#8a7a68" 
              tick={{ fill: '#8a7a68', fontSize: 11 }}
              label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#8a7a68', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#8a7a68', strokeWidth: 1 }} />
            <Line 
              type="linear" 
              dataKey="randomTpr" 
              stroke="#6b5d4f" 
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              isAnimationActive={false}
            />
            <Line 
              type="monotone" 
              dataKey="tpr" 
              stroke="#d4a574" 
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 6, fill: '#f5f1ed', stroke: '#d4a574', strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-[#8a7a68] text-center mt-2 font-light">
          The curve&apos;s distance from the diagonal indicates strong predictive power
        </p>
      </div>
    </div>
  );
}
