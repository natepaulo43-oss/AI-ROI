import Link from 'next/link';

export default function Header() {
  return (
    <header className="border-b border-[#5a4d3f]/40">
      <nav className="mx-auto max-w-6xl px-8 py-6">
        <div className="flex items-baseline justify-between">
          <Link href="/" className="text-sm font-normal text-[#f5f1ed] tracking-wide uppercase">
            AI ROI Research Tool
          </Link>
          <div className="flex gap-10">
            <Link href="/tool" className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors">
              Tool
            </Link>
            <Link href="/hypothesis" className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors">
              Motivations
            </Link>
            <Link href="/methodology" className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors">
              Methodology
            </Link>
            <Link href="/insights" className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors">
              Insights
            </Link>
            <Link href="/limitations" className="text-sm font-normal text-[#c4b5a0] hover:text-[#f5f1ed] transition-colors">
              Limitations
            </Link>
          </div>
        </div>
      </nav>
    </header>
  );
}
