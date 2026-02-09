interface InterpretationPanelProps {
  interpretation: string;
}

export default function InterpretationPanel({ interpretation }: InterpretationPanelProps) {
  return (
    <div className="mt-12">
      <h3 className="text-xs uppercase tracking-widest text-stone-500 mb-4 font-normal">Interpretation</h3>
      <p className="text-stone-700 leading-relaxed font-light mb-8">{interpretation}</p>
      <div className="border-l-2 border-stone-300 pl-6 py-2">
        <p className="text-sm text-stone-600 leading-relaxed font-light">
          This prediction is an estimate based on historical data and machine learning models. 
          It should be used as a decision-support tool, not as a guarantee of actual outcomes. 
          Human judgment and domain expertise remain essential.
        </p>
      </div>
    </div>
  );
}
