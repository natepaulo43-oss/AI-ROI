interface FeatureItemProps {
  number: string;
  title: string;
  description: string;
}

function FeatureItem({ number, title, description }: FeatureItemProps) {
  return (
    <div className="grid grid-cols-12 gap-8 items-start">
      <div className="col-span-1 text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68]">
        {number}
      </div>
      <div className="col-span-3">
        <h3 className="text-2xl font-light text-[#f5f1ed] leading-tight">{title}</h3>
      </div>
      <div className="col-span-7">
        <p className="text-[#e8dfd5] leading-relaxed font-light text-sm">{description}</p>
      </div>
    </div>
  );
}

export default function ValueProposition() {
  return (
    <div className="mt-32 space-y-12">
      <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-16">
        Key Capabilities
      </div>
      
      <FeatureItem
        number="01"
        title="Data-Driven Predictions"
        description="Built on empirical research and real-world data from SME AI adoption initiatives across multiple industries and firm sizes"
      />
      <FeatureItem
        number="02"
        title="Machine Learning Core"
        description="Powered by gradient boosting regression models trained on firm characteristics, operational maturity, and historical outcomes"
      />
      <FeatureItem
        number="03"
        title="SME-Focused Design"
        description="Specifically designed for small and medium enterprises navigating AI investment decisions with limited resources"
      />
    </div>
  );
}
