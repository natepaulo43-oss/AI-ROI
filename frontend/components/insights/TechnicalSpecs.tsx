interface Metric {
  label: string;
  value: string;
  description: string;
}

interface TechnicalSpecsProps {
  metrics: Metric[];
}

export default function TechnicalSpecs({ metrics }: TechnicalSpecsProps) {
  return (
    <div className="max-w-3xl">
      <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Performance Metrics</h2>
      <div className="space-y-6">
        {metrics.map((metric, index) => (
          <div key={index} className="flex items-baseline justify-between border-b border-[#5a4d3f]/40 pb-4">
            <div className="flex-1">
              <p className="text-sm font-normal text-[#f5f1ed] mb-1">
                {metric.label}
              </p>
              <p className="text-xs text-[#b8a894] font-light">{metric.description}</p>
            </div>
            <p className="text-3xl font-light text-[#f5f1ed] tabular-nums ml-8">{metric.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
