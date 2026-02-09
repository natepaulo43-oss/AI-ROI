import Link from 'next/link';

export default function CTAButton() {
  return (
    <div className="mt-24 grid grid-cols-12 gap-8">
      <div className="col-span-1"></div>
      <div className="col-span-11">
        <Link 
          href="/tool" 
          className="inline-flex items-center gap-3 text-[#f5f1ed] hover:text-[#e8dfd5] transition-colors group"
        >
          <span className="text-[0.7rem] uppercase tracking-[0.15em] font-normal">Explore Tool</span>
          <span className="w-8 h-8 rounded-full border border-[#b8a894] flex items-center justify-center group-hover:border-[#f5f1ed] transition-colors">
            <span className="text-sm">→</span>
          </span>
        </Link>
      </div>
    </div>
  );
}
