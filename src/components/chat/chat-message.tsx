import { AlertCircle, Info, RotateCcw } from "lucide-react";
import type { AskResponse } from "@/lib/ask-types";
import { AnswerMarkdown } from "./answer-markdown";

export type ChatEntry =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; data: AskResponse }
  | { id: string; role: "error"; status: number | null; question: string };

function friendlyErrorMessage(status: number | null): string {
  if (status === 422) {
    return "The question couldn't be processed by the service. Try rephrasing it and sending again.";
  }
  // 500 / 502 / 503 / network / cold start
  return "The AI service is waking up or temporarily unavailable. Please try again in a moment.";
}

function RouteBadge({
  route,
  reason,
}: {
  route: string;
  reason?: string | undefined;
}) {
  const isVisual = route === "VISUAL_REQUIRED";
  return (
    <span
      title={reason || undefined}
      className={
        isVisual
          ? "inline-flex items-center rounded-full border border-highlight-foreground/30 bg-highlight/40 px-2.5 py-0.5 font-mono text-[11px] font-medium tracking-wide text-highlight-foreground"
          : "inline-flex items-center rounded-full border border-border bg-secondary px-2.5 py-0.5 font-mono text-[11px] font-medium tracking-wide text-secondary-foreground"
      }
    >
      {route}
    </span>
  );
}

function AnswerMetadata({
  data,
  onViewPage,
}: {
  data: AskResponse;
  onViewPage: (page: number) => void;
}) {
  const pages = Array.from(new Set(data.sources.map((s) => s.page)));
  const latencySec = (data.latency_ms.total / 1000).toFixed(1);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <span className="font-medium text-foreground/70">Route</span>
        <RouteBadge route={data.route} reason={data.routing_reason} />
      </span>
      <span>
        <span className="font-medium text-foreground/70">Vision</span>{" "}
        {data.vision_used ? "Used" : "Not used"}
      </span>
      <span>
        <span className="font-medium text-foreground/70">Fallback</span>{" "}
        {data.fallback_used ? "Yes" : "No"}
      </span>
      <span>
        <span className="font-medium text-foreground/70">Latency</span>{" "}
        <span className="font-mono">{latencySec}s</span>
      </span>
      {pages.length > 0 && (
        <span className="flex flex-wrap items-center gap-1.5">
          <span className="font-medium text-foreground/70">Sources</span>
          {pages.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => onViewPage(p)}
              className="rounded-md border border-border bg-card px-1.5 py-0.5 font-mono text-[11px] text-foreground/80 transition-colors hover:border-ring/50 hover:bg-accent hover:text-foreground"
            >
              Page {p}
            </button>
          ))}
        </span>
      )}
    </div>
  );
}

function AssistantMessage({
  data,
  onViewPage,
}: {
  data: AskResponse;
  onViewPage: (page: number) => void;
}) {
  const visualFallback =
    data.route === "VISUAL_REQUIRED" && !data.vision_used && data.fallback_used;

  return (
    <article className="animate-fade-up overflow-hidden rounded-xl border border-border bg-card shadow-[0_1px_2px_oklch(0.3_0.02_262/0.05)]">
      <div className="px-5 py-4 sm:px-6 sm:py-5">
        <AnswerMarkdown content={data.answer} />
      </div>

      {visualFallback && (
        <div className="mx-5 mb-4 flex items-start gap-2.5 rounded-lg border border-highlight-foreground/25 bg-highlight/25 px-3.5 py-3 sm:mx-6">
          <Info className="mt-0.5 size-4 shrink-0 text-highlight-foreground" />
          <p className="text-xs leading-5 text-highlight-foreground">
            Visual reasoning was requested, but the web demo is currently running
            in hosted-only mode. The answer was generated using text RAG
            fallback.
          </p>
        </div>
      )}

      <footer className="border-t border-border bg-muted/60 px-5 py-3 sm:px-6">
        <AnswerMetadata data={data} onViewPage={onViewPage} />
      </footer>
    </article>
  );
}

function ErrorMessage({
  status,
  question,
  onRetry,
}: {
  status: number | null;
  question: string;
  onRetry: (question: string) => void;
}) {
  return (
    <div className="animate-fade-up rounded-xl border border-destructive/30 bg-destructive/5 px-5 py-4">
      <div className="flex items-start gap-2.5">
        <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
        <div className="flex-1">
          <p className="text-sm leading-6 text-foreground">
            {friendlyErrorMessage(status)}
          </p>
          <button
            type="button"
            onClick={() => onRetry(question)}
            className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1 text-xs font-medium text-foreground transition-colors hover:bg-accent"
          >
            <RotateCcw className="size-3" />
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}

export function ChatEntryView({
  entry,
  onRetry,
  onViewPage,
}: {
  entry: ChatEntry;
  onRetry: (question: string) => void;
  onViewPage: (page: number) => void;
}) {
  if (entry.role === "user") {
    return (
      <div className="animate-fade-up flex justify-end">
        <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-[15px] leading-6 text-primary-foreground sm:max-w-[75%]">
          {entry.text}
        </p>
      </div>
    );
  }

  if (entry.role === "assistant") {
    return <AssistantMessage data={entry.data} onViewPage={onViewPage} />;
  }

  return (
    <ErrorMessage
      status={entry.status}
      question={entry.question}
      onRetry={onRetry}
    />
  );
}
