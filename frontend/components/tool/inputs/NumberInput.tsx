interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  placeholder?: string;
}

export default function NumberInput({
  label,
  value,
  onChange,
  step = 1,
  placeholder,
}: NumberInputProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value;
    
    // Allow empty string (user is clearing the field)
    if (inputValue === '') {
      onChange(0);
      return;
    }
    
    // Parse the number
    const numValue = Number(inputValue);
    
    // Only update if it's a valid number and non-negative
    if (!isNaN(numValue) && numValue >= 0) {
      onChange(numValue);
    }
  };

  // Prevent scroll wheel from changing the value
  const handleWheel = (e: React.WheelEvent<HTMLInputElement>) => {
    e.currentTarget.blur();
  };

  return (
    <div className="mb-8">
      <label className="label">{label}</label>
      <input
        type="number"
        className="input-field text-[#f5f1ed] [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        value={value === 0 ? '' : value}
        onChange={handleChange}
        onWheel={handleWheel}
        min="0"
        placeholder={placeholder || '0'}
      />
    </div>
  );
}
