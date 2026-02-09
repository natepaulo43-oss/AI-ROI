interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
}

export default function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  placeholder,
}: NumberInputProps) {
  return (
    <div className="mb-8">
      <label className="label">{label}</label>
      <input
        type="number"
        className="input-field text-[#f5f1ed]"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
      />
    </div>
  );
}
