'use client';

interface ConfusionMatrixChartProps {
  data?: {
    truePositive: number;
    falsePositive: number;
    trueNegative: number;
    falseNegative: number;
  };
}

export default function ConfusionMatrixChart({ data }: ConfusionMatrixChartProps) {
  const matrix = data || {
    truePositive: 25,
    falsePositive: 9,
    trueNegative: 54,
    falseNegative: 15,
  };

  const total = matrix.truePositive + matrix.falsePositive + matrix.trueNegative + matrix.falseNegative;
  const accuracy = ((matrix.truePositive + matrix.trueNegative) / total * 100).toFixed(1);
  const precision = (matrix.truePositive / (matrix.truePositive + matrix.falsePositive) * 100).toFixed(1);
  const recall = (matrix.truePositive / (matrix.truePositive + matrix.falseNegative) * 100).toFixed(1);
  const specificity = (matrix.trueNegative / (matrix.trueNegative + matrix.falsePositive) * 100).toFixed(1);

  const maxValue = Math.max(matrix.truePositive, matrix.falsePositive, matrix.trueNegative, matrix.falseNegative);

  const getCellOpacity = (value: number) => {
    return 0.3 + (value / maxValue) * 0.7;
  };

  const getCellColor = (isCorrect: boolean) => {
    return isCorrect ? '#d4a574' : '#6b5d4f';
  };

  return (
    <div className="grid grid-cols-12 gap-12">
      <div className="col-span-4">
        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-3">Figure 6</div>
        <h3 className="text-3xl font-light text-[#f5f1ed] mb-6 leading-tight">
          Confusion Matrix
        </h3>
        <p className="text-sm text-[#e8dfd5] font-light leading-relaxed mb-6">
          Classification performance breakdown showing the model correctly identifies both High ROI 
          and Not-High ROI projects, avoiding majority-class bias.
        </p>
        <div className="space-y-3 text-xs font-light">
          <div>
            <div className="text-[#f5f1ed] font-normal mb-1">Precision (High ROI)</div>
            <div className="text-[#d4a574] text-lg font-light">{precision}%</div>
            <div className="text-[#b8a894]">When predicting High, {precision}% are correct</div>
          </div>
          <div>
            <div className="text-[#f5f1ed] font-normal mb-1">Recall (High ROI)</div>
            <div className="text-[#d4a574] text-lg font-light">{recall}%</div>
            <div className="text-[#b8a894]">Catches {recall}% of actual High ROI cases</div>
          </div>
          <div>
            <div className="text-[#f5f1ed] font-normal mb-1">Specificity (Not-High)</div>
            <div className="text-[#d4a574] text-lg font-light">{specificity}%</div>
            <div className="text-[#b8a894]">Correctly identifies {specificity}% of Not-High</div>
          </div>
        </div>
      </div>

      <div className="col-span-7 col-start-6">
        <div className="flex flex-col items-center pt-8">
          {/* Matrix Grid */}
          <div className="relative">
            {/* Column Labels */}
            <div className="flex mb-4 ml-32">
              <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] mb-2">
                Predicted
              </div>
            </div>
            <div className="flex mb-2 ml-32">
              <div className="w-40 text-center text-xs text-[#e8dfd5] font-light">Not-High ROI</div>
              <div className="w-40 text-center text-xs text-[#e8dfd5] font-light">High ROI</div>
            </div>

            {/* Matrix with Row Labels */}
            <div className="flex">
              {/* Row Labels */}
              <div className="flex flex-col justify-center mr-4">
                <div className="text-[0.65rem] uppercase tracking-[0.2em] text-[#8a7a68] -rotate-90 whitespace-nowrap mb-8">
                  Actual
                </div>
              </div>
              <div className="flex flex-col justify-center mr-4">
                <div className="h-32 flex items-center">
                  <div className="text-xs text-[#e8dfd5] font-light whitespace-nowrap">Not-High ROI</div>
                </div>
                <div className="h-32 flex items-center">
                  <div className="text-xs text-[#e8dfd5] font-light whitespace-nowrap">High ROI</div>
                </div>
              </div>

              {/* Matrix Cells */}
              <div className="grid grid-cols-2 gap-3">
                {/* True Negative */}
                <div 
                  className="w-40 h-32 rounded-lg flex flex-col items-center justify-center border border-[#4a3f35]"
                  style={{ 
                    backgroundColor: getCellColor(true),
                    opacity: getCellOpacity(matrix.trueNegative)
                  }}
                >
                  <div className="text-3xl font-light text-[#f5f1ed] mb-1">{matrix.trueNegative}</div>
                  <div className="text-[0.6rem] uppercase tracking-wider text-[#e8dfd5]">True Negative</div>
                  <div className="text-xs text-[#b8a894] mt-1">{(matrix.trueNegative / total * 100).toFixed(1)}%</div>
                </div>

                {/* False Positive */}
                <div 
                  className="w-40 h-32 rounded-lg flex flex-col items-center justify-center border border-[#4a3f35]"
                  style={{ 
                    backgroundColor: getCellColor(false),
                    opacity: getCellOpacity(matrix.falsePositive)
                  }}
                >
                  <div className="text-3xl font-light text-[#f5f1ed] mb-1">{matrix.falsePositive}</div>
                  <div className="text-[0.6rem] uppercase tracking-wider text-[#e8dfd5]">False Positive</div>
                  <div className="text-xs text-[#b8a894] mt-1">{(matrix.falsePositive / total * 100).toFixed(1)}%</div>
                </div>

                {/* False Negative */}
                <div 
                  className="w-40 h-32 rounded-lg flex flex-col items-center justify-center border border-[#4a3f35]"
                  style={{ 
                    backgroundColor: getCellColor(false),
                    opacity: getCellOpacity(matrix.falseNegative)
                  }}
                >
                  <div className="text-3xl font-light text-[#f5f1ed] mb-1">{matrix.falseNegative}</div>
                  <div className="text-[0.6rem] uppercase tracking-wider text-[#e8dfd5]">False Negative</div>
                  <div className="text-xs text-[#b8a894] mt-1">{(matrix.falseNegative / total * 100).toFixed(1)}%</div>
                </div>

                {/* True Positive */}
                <div 
                  className="w-40 h-32 rounded-lg flex flex-col items-center justify-center border border-[#4a3f35]"
                  style={{ 
                    backgroundColor: getCellColor(true),
                    opacity: getCellOpacity(matrix.truePositive)
                  }}
                >
                  <div className="text-3xl font-light text-[#f5f1ed] mb-1">{matrix.truePositive}</div>
                  <div className="text-[0.6rem] uppercase tracking-wider text-[#e8dfd5]">True Positive</div>
                  <div className="text-xs text-[#b8a894] mt-1">{(matrix.truePositive / total * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          </div>

          <p className="text-xs text-[#8a7a68] text-center mt-6 font-light max-w-lg">
            Model achieves {accuracy}% overall accuracy with balanced performance across both classes
          </p>
        </div>
      </div>
    </div>
  );
}
