import { useEffect, useState } from "react";

const PHASES = [
  "Analyzing the paper…",
  "Routing the question…",
  "Retrieving relevant passages…",
  "Generating the answer…",
];

const COLD_START_HINT =
  "The hosted service is waking up from a cold start — this can take up to a minute.";

export function TypingIndicator() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const label =
    elapsed >= 20 ? COLD_START_HINT : PHASES[Math.floor(elapsed / 4) % PHASES.length];

  return (
    <div className="animate-fade-up rounded-xl border border-border bg-card px-5 py-4">
      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1" aria-hidden="true">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="animate-typing-dot size-1.5 rounded-full bg-muted-foreground"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </span>
        <p className="text-sm text-muted-foreground" role="status">
          {label}
        </p>
      </div>
    </div>
  );
}
