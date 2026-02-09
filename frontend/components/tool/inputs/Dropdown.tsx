interface DropdownProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}

export default function Dropdown({ label, value, onChange, options }: DropdownProps) {
  return (
    <div className="mb-8">
      <label className="label">{label}</label>
      <select
        className="input-field text-[#f5f1ed] cursor-pointer bg-[#2a2520] appearance-none pr-8"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23b8a894' d='M6 9L1 4h10z'/%3E%3C/svg%3E")`,
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'right 0.5rem center',
        }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="" style={{ backgroundColor: '#3d342a', color: '#e8dfd5' }}>—</option>
        {options.map((option) => (
          <option 
            key={option.value} 
            value={option.value}
            style={{ backgroundColor: '#3d342a', color: '#e8dfd5' }}
          >
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
