import { ReactNode } from 'react';

interface FormSectionProps {
  title: string;
  children: ReactNode;
}

export default function FormSection({ title, children }: FormSectionProps) {
  return (
    <div className="mb-12">
      <h3 className="text-xs uppercase tracking-widest text-[#8a7a68] mb-6 font-normal">
        {title}
      </h3>
      {children}
    </div>
  );
}
