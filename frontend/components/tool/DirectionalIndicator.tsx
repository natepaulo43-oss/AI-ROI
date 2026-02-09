interface DirectionalIndicatorProps {
  direction: 'positive' | 'neutral' | 'negative';
}

export default function DirectionalIndicator({ direction }: DirectionalIndicatorProps) {
  const config = {
    positive: {
      icon: '↑',
      label: 'Positive',
    },
    neutral: {
      icon: '→',
      label: 'Neutral',
    },
    negative: {
      icon: '↓',
      label: 'Negative',
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
