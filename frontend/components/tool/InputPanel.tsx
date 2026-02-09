import { ReactNode } from 'react';

interface InputPanelProps {
  children: ReactNode;
  onSubmit: () => void;
  isLoading: boolean;
}

export default function InputPanel({ children, onSubmit, isLoading }: InputPanelProps) {
  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
      >
        {children}
        <button
          type="submit"
          className="btn-primary w-full mt-8"
          disabled={isLoading}
        >
          {isLoading ? 'Calculating...' : 'Calculate Prediction'}
        </button>
      </form>
    </div>
  );
}
