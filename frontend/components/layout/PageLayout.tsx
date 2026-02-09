import { ReactNode } from 'react';

interface PageLayoutProps {
  children: ReactNode;
  title?: string;
  description?: string;
}

export default function PageLayout({ children, title, description }: PageLayoutProps) {
  return (
    <div className="mx-auto max-w-6xl px-8 py-16">
      {(title || description) && (
        <div className="mb-16 max-w-3xl">
          {title && (
            <h1 className="text-5xl font-light text-stone-900 tracking-tight leading-tight mb-6">
              {title}
            </h1>
          )}
          {description && (
            <p className="text-lg text-stone-600 leading-relaxed font-light">
              {description}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
