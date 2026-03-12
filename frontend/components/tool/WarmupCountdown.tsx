'use client';

import { useEffect, useState } from 'react';

interface WarmupCountdownProps {
  onRetry: () => void;
  initialSeconds?: number;
  errorMessage?: string;
}

export default function WarmupCountdown({ 
  onRetry, 
  initialSeconds = 60,
  errorMessage 
}: WarmupCountdownProps) {
  const [secondsLeft, setSecondsLeft] = useState(initialSeconds);
  const [isRetrying, setIsRetrying] = useState(false);

  useEffect(() => {
    if (secondsLeft <= 0) {
      setIsRetrying(true);
      onRetry();
      return;
    }

    const timer = setInterval(() => {
      setSecondsLeft((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [secondsLeft, onRetry]);

  const progress = ((initialSeconds - secondsLeft) / initialSeconds) * 100;

  return (
    <div className="bg-amber-900/20 border border-amber-500/30 rounded-[2rem] p-8">
      <div className="flex items-start gap-4 mb-6">
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
          <svg 
            className="w-5 h-5 text-amber-400 animate-pulse" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M13 10V3L4 14h7v7l9-11h-7z" 
            />
          </svg>
        </div>
        <div className="flex-1">
          <h3 className="text-amber-300 font-medium text-base mb-2">
            {isRetrying ? 'Retrying...' : 'API Warming Up'}
          </h3>
          <p className="text-amber-200/80 text-sm leading-relaxed mb-4">
            {errorMessage || 'The API is starting up from sleep mode. This happens on the first request and typically takes 30-60 seconds.'}
          </p>
          
          {!isRetrying && (
            <>
              <div className="flex items-center gap-3 mb-3">
                <div className="text-3xl font-light text-amber-300 tabular-nums">
                  {secondsLeft}
                </div>
                <div className="text-sm text-amber-400/60">
                  second{secondsLeft !== 1 ? 's' : ''} until auto-retry
                </div>
              </div>
              
              <div className="relative w-full h-1.5 bg-amber-950/50 rounded-full overflow-hidden">
                <div 
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-1000 ease-linear rounded-full"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </>
          )}

          {isRetrying && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-amber-300">Attempting to connect...</span>
            </div>
          )}
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => {
            setIsRetrying(true);
            onRetry();
          }}
          disabled={isRetrying}
          className="px-4 py-2 text-xs uppercase tracking-widest bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Retry Now
        </button>
        <div className="text-xs text-amber-400/40 flex items-center">
          Free tier hosting • First request may be slow
        </div>
      </div>
    </div>
  );
}
