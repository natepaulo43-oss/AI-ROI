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
        description="Trained on 514 real-world AI deployment cases including Fortune 500 implementations (Klarna, Alibaba, JPMorgan) across 16 sectors and 15 use cases"
      />
      <FeatureItem
        number="02"
        title="Binary Classification Model"
        description="XGBoost classifier predicting High vs Not-High ROI (≥145.5% threshold) with 86.4% accuracy, significantly outperforming regression approaches"
      />
      <FeatureItem
        number="03"
        title="SME-Focused Design"
        description="Specifically designed for small and medium enterprises navigating AI investment decisions with limited resources"
      />
    </div>
  );
}
