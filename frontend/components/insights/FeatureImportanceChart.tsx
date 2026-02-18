interface FeatureImportance {
  feature: string;
  importance: number;
}

interface FeatureImportanceChartProps {
  data: FeatureImportance[];
}

export default function FeatureImportanceChart({ data }: FeatureImportanceChartProps) {
  const maxImportance = Math.max(...data.map((d) => d.importance));

  return (
    <div className="grid grid-cols-12 gap-12">
      {/* Left: Title and description */}
      <div className="col-span-4">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-3">Figure 4</div>
        <h3 className="text-3xl font-light text-[#f5f1ed] mb-6 leading-tight">
          Feature
          <br />
          Importance
        </h3>
        <p className="text-sm text-[#e8dfd5] font-light leading-relaxed">
          Features ranked by their contribution to the model&apos;s predictions. Higher values indicate 
          greater influence on ROI outcomes.
        </p>
      </div>

      {/* Right: Chart data */}
      <div className="col-span-7 col-start-6">
        <div className="space-y-6 pt-8">
          {data.map((item, index) => (
            <div key={index}>
              <div className="flex justify-between items-baseline mb-2">
                <span className="text-sm font-light text-[#e8dfd5]">{item.feature}</span>
                <span className="text-xs font-light text-[#b8a894] tabular-nums">
                  {(item.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-[#4a3f35] h-1">
                <div
                  className="bg-[#e8dfd5] h-1"
                  style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
