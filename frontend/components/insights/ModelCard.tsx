export default function ModelCard() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">Model Overview</h2>
      <div className="space-y-8">
        <div>
          <h3 className="text-2xl font-light text-[#f5f1ed] mb-3">Binary Classification (XGBoost)</h3>
          <p className="text-[#e8dfd5] leading-relaxed font-light">
            XGBoost binary classifier predicting High ROI (above 145.5%) vs Not-High ROI projects. 
            Uses 500 trees with depth 8 and learning rate 0.03. Achieves 68.82% accuracy with 
            70.76% AUC-ROC, significantly outperforming regression approaches (329% improvement). 
            Statistically significant with p less than 0.001.
          </p>
        </div>
        
        <div className="border-l-2 border-[#6b5d4f] pl-6 py-2">
          <h3 className="text-sm font-normal text-[#f5f1ed] mb-3">Why Binary Classification?</h3>
          <ul className="text-[#e8dfd5] leading-relaxed font-light space-y-2">
            <li>— Regression achieved only 16% R² (unusable)</li>
            <li>— 3-class classification reached 51.6% accuracy</li>
            <li>— Binary approach achieves 68.8% accuracy (33% improvement)</li>
            <li>— Clearer business decision: High vs Not-High ROI</li>
            <li>— Handles class imbalance with scale_pos_weight=2</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
