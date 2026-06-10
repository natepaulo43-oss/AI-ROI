'use client';
import Link from 'next/link';
import { useState } from 'react';

const NAV_LINKS = [
  { href: '/tool', label: 'Tool' },
  { href: '/hypothesis', label: 'Motivations' },
  { href: '/methodology', label: 'Methodology' },
  { href: '/insights', label: 'Insights' },
  { href: '/limitations', label: 'Limitations' },
];

export default function Header() {
  const [open, setOpen] = useState(false);

  return (
    <header className="border-b border-[#5a4d3f]/40">
      <nav className="mx-auto max-w-6xl px-4 sm:px-8 py-6">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="text-sm font-normal text-[#f5f1ed] tracking-wide uppercase"
            onClick={() => setOpen(false)}
          >
            AI ROI Research Tool
          </Link>

          {/* Desktop nav */}
          <div className="hidden sm:flex gap-10">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors"
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Mobile hamburger */}
          <button
            className="sm:hidden text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors p-1"
            onClick={() => setOpen(!open)}
            aria-label="Toggle navigation menu"
          >
            {open ? (
              <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
              </svg>
            )}
          </button>
        </div>

        {/* Mobile dropdown */}
        {open && (
          <div className="sm:hidden flex flex-col gap-1 mt-4 pt-4 border-t border-[#5a4d3f]/40">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors py-2"
                onClick={() => setOpen(false)}
              >
                {link.label}
              </Link>
            ))}
          </div>
        )}
      </nav>
    </header>
  );
}
