interface DropdownProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; group?: string }[];
}

export default function Dropdown({ label, value, onChange, options }: DropdownProps) {
  // Group options by their group property
  const hasGroups = options.some(opt => opt.group);
  
  const renderGroupedOptions = () => {
    if (!hasGroups) {
      return options.map((option) => (
        <option 
          key={option.value} 
          value={option.value}
          style={{ backgroundColor: '#3d342a', color: '#e8dfd5' }}
        >
          {option.label}
        </option>
      ));
    }

    const groups: { [key: string]: typeof options } = {};
    options.forEach(option => {
      const groupName = option.group || 'Other';
      if (!groups[groupName]) groups[groupName] = [];
      groups[groupName].push(option);
    });

    return Object.entries(groups).map(([groupName, groupOptions]) => (
      <optgroup 
        key={groupName} 
        label={groupName}
        style={{ backgroundColor: '#2a2520', color: '#b8a894', fontWeight: 'bold' }}
      >
        {groupOptions.map((option) => (
          <option 
            key={option.value} 
            value={option.value}
            style={{ backgroundColor: '#3d342a', color: '#e8dfd5', paddingLeft: '1rem' }}
          >
            {option.label}
          </option>
        ))}
      </optgroup>
    ));
  };

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
        <option value="" style={{ backgroundColor: '#3d342a', color: '#8a7a68' }}>Select an option...</option>
        {renderGroupedOptions()}
      </select>
    </div>
  );
}
