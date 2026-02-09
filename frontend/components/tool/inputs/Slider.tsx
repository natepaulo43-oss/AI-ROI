interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
}

export default function Slider({ label, value, onChange, min, max, step = 1 }: SliderProps) {
  return (
    <div className="mb-8">
      <label className="label mb-4">
        {label}: <span className="font-normal text-[#f5f1ed]">{value}</span>
      </label>
      <input
        type="range"
        className="w-full h-1 bg-[#4a3f35] appearance-none cursor-pointer accent-[#e8dfd5]"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
      <div className="flex justify-between text-xs text-[#8a7a68] mt-2 font-light">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
