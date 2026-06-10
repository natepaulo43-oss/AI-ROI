interface FeatureItemProps {
  number: string;
  title: string;
  description: string;
}

function FeatureItem({ number, title, description }: FeatureItemProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 sm:gap-8 items-start">
      <div className="hidden sm:block sm:col-span-1 text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68]">
        {number}
      </div>
      <div className="sm:col-span-3">
        <div className="flex items-baseline gap-3 sm:block">
          <span className="sm:hidden text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68]">{number}</span>
          <h3 className="text-xl sm:text-2xl font-light text-[#f5f1ed] leading-tight">{title}</h3>
        </div>
      </div>
      <div className="sm:col-span-7">
        <p className="text-[#e8dfd5] leading-relaxed font-light text-sm">{description}</p>
      </div>
    </div>
  );
}

export default function ValueProposition() {
  return (
    <div className="mt-16 sm:mt-32 space-y-10 sm:space-y-12">
      <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-10 sm:mb-16">
        Key Capabilities
      </div>

      <FeatureItem
        number="01"
        title="Data-Driven Predictions"
        description="Trained on 514 real-world AI deployment cases across 16 sectors and 15 use cases"
      />
      <FeatureItem
        number="02"
        title="Binary Classification Model"
        description="Gradient Boosting classifier predicting High vs Not-High ROI (≥145.5% threshold) with 76.7% accuracy and 75.5% average confidence, significantly outperforming regression approaches"
      />
      <FeatureItem
        number="03"
        title="SME-Focused Design"
        description="Specifically designed for small and medium enterprises navigating AI investment decisions with limited resources"
      />
    </div>
  );
}
