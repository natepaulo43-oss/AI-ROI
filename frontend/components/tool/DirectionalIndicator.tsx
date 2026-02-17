interface DirectionalIndicatorProps {
  direction: 'high' | 'not-high';
}

export default function DirectionalIndicator({ direction }: DirectionalIndicatorProps) {
  const config = {
    high: {
      icon: '↑',
      label: 'High ROI',
    },
    'not-high': {
      icon: '→',
      label: 'Not-High ROI',
    },
  };

  const { icon, label } = config[direction];

  return (
    <div className="inline-flex items-center gap-3 text-[#e8dfd5]">
      <span className="text-3xl font-light">{icon}</span>
      <span className="text-sm uppercase tracking-widest font-normal">{label}</span>
    </div>
  );
}
