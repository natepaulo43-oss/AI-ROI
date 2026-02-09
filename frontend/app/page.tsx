import Hero from '@/components/landing/Hero';
import ValueProposition from '@/components/landing/ValueProposition';
import CTAButton from '@/components/landing/CTAButton';

export default function Home() {
  return (
    <div className="mx-auto max-w-7xl px-12 py-8">
      <Hero />
      <ValueProposition />
      <CTAButton />
    </div>
  );
}
