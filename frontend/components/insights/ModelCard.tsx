export default function ModelCard() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Model Overview</h2>
      <div className="space-y-8">
        <div>
          <h3 className="text-2xl font-light text-[#f5f1ed] mb-3">Binary Classification (Gradient Boosting)</h3>
          <p className="text-[#e8dfd5] leading-relaxed font-light">
            Gradient Boosting binary classifier predicting High ROI (≥145.5%) vs Not-High ROI projects. 
            Uses 500 trees with depth 5 and learning rate 0.05. Achieves 76.70% accuracy with 
            76.74% AUC-ROC and 75.5% average confidence, significantly outperforming regression approaches. 
            Validated with 5-fold cross-validation.
          </p>
        </div>
        
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h3 className="text-sm font-normal text-[#f5f1ed] mb-3">Why Binary Classification?</h3>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Regression achieved only 16% R² (unusable)</li>
            <li>— 3-class classification reached 51.6% accuracy</li>
            <li>— Binary approach achieves 76.7% accuracy (best achievable)</li>
            <li>— Clearer business decision: High vs Not-High ROI</li>
            <li>— Handles class imbalance with scale_pos_weight=1.87</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
