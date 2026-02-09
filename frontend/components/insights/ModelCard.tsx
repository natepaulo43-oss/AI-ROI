export default function ModelCard() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Model Overview</h2>
      <div className="space-y-8">
        <div>
          <h3 className="text-2xl font-light text-[#f5f1ed] mb-3">Gradient Boosting Regression (XGBoost)</h3>
          <p className="text-[#e8dfd5] leading-relaxed font-light">
            XGBoost is an ensemble learning method that builds multiple decision trees sequentially, 
            with each tree correcting errors from previous trees. It is particularly effective for 
            tabular data and provides excellent predictive performance while maintaining interpretability 
            through feature importance metrics.
          </p>
        </div>
        
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h3 className="text-sm font-normal text-[#f5f1ed] mb-3">Rationale</h3>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Handles mixed categorical and numerical features effectively</li>
            <li>— Robust to outliers and missing data</li>
            <li>— Provides feature importance for transparency</li>
            <li>— Strong performance on SME business metrics</li>
            <li>— Widely validated in academic and industry research</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
