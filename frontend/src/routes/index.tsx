import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpenText, FlaskConical } from "lucide-react";
import { askPaperQuestion, getBackendHealth } from "@/lib/ask.functions";
import { ChatEntryView, type ChatEntry } from "@/components/chat/chat-message";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { PageViewer } from "@/components/chat/page-viewer";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Furiosa Agentic PDF RAG — Selective Multimodal RAG Demo" },
      {
        name: "description",
        content:
          'Ask questions about "Attention Is All You Need". The agentic RAG system adaptively decides whether visual reasoning is required.',
      },
      {
        property: "og:title",
        content: "Furiosa Agentic PDF RAG — Selective Multimodal RAG Demo",
      },
      {
        property: "og:description",
        content:
          'Interactive research demo: ask questions about "Attention Is All You Need" with adaptive text/visual routing.',
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: DemoPage,
});

const SUGGESTED_QUESTIONS = [
  "왜 multi-head attention을 사용하는가?",
  "scaled dot-product attention에서 sqrt(d_k)로 나누는 이유는?",
  "Figure 1에서 Encoder와 Decoder 구조의 차이는?",
  "Encoder의 출력은 Decoder의 어느 attention block으로 연결되는가?",
];

type BackendStatus = "connecting" | "online" | "offline";

function DemoPage() {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [backendStatus, setBackendStatus] =
    useState<BackendStatus>("connecting");
  const [viewerPage, setViewerPage] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [entries, pending]);

  // Warm-up ping on load: wakes a sleeping Render instance before the
  // first question and drives the header status indicator.
  useEffect(() => {
    let cancelled = false;
    void getBackendHealth().then((res) => {
      if (!cancelled) setBackendStatus(res.ok ? "online" : "offline");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const openViewer = useCallback((page: number) => setViewerPage(page), []);
  const closeViewer = useCallback(() => setViewerPage(null), []);

  async function send(rawQuestion: string) {
    const question = rawQuestion.trim();
    if (!question || pending) return;

    setInput("");
    setPending(true);
    setEntries((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", text: question },
    ]);

    const result = await askPaperQuestion({ data: { question } });

    setEntries((prev) => [
      ...prev,
      result.ok
        ? { id: crypto.randomUUID(), role: "assistant", data: result.data }
        : {
            id: crypto.randomUUID(),
            role: "error",
            status: result.status,
            question,
          },
    ]);
    if (result.ok) setBackendStatus("online");
    setPending(false);
    inputRef.current?.focus();
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void send(input);
  }

  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="border-b border-border bg-background">
        <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <div className="min-w-0">
            <h1 className="truncate font-serif text-lg font-semibold tracking-tight text-foreground">
              Furiosa Agentic PDF RAG
            </h1>
            <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Selective Multimodal RAG Demo
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <StatusPill status={backendStatus} />
            <button
              type="button"
              onClick={() => openViewer(1)}
              title="Browse the paper page by page"
              className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-ring/50 hover:bg-accent hover:text-foreground"
            >
              <BookOpenText className="size-3.5" />
              <span className="hidden font-serif italic sm:inline">
                Attention Is All You Need
              </span>
            </button>
          </div>
        </div>
      </header>

      <main ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-8 sm:px-6">
          {entries.length === 0 && !pending ? (
            <EmptyState onAsk={(q) => void send(q)} />
          ) : (
            entries.map((entry) => (
              <ChatEntryView
                key={entry.id}
                entry={entry}
                onRetry={(q) => void send(q)}
                onViewPage={openViewer}
              />
            ))
          )}
          {pending && <TypingIndicator />}
        </div>
      </main>

      <footer className="border-t border-border bg-background">
        <form
          onSubmit={handleSubmit}
          className="mx-auto w-full max-w-3xl px-4 py-4 sm:px-6"
        >
          <div className="flex items-end gap-2 rounded-2xl border border-input bg-card p-2 shadow-[0_1px_3px_oklch(0.3_0.02_262/0.06)] focus-within:border-ring/60">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send(input);
                }
              }}
              placeholder="Ask a question about the paper…"
              rows={1}
              className="field-sizing-content max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-6 text-foreground outline-none placeholder:text-muted-foreground"
            />
            <button
              type="submit"
              disabled={pending || !input.trim()}
              aria-label="Send question"
              className="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-35"
            >
              <ArrowUp className="size-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Responses are generated by the hosted research backend and may take
            up to a minute on cold start.
          </p>
        </form>
      </footer>

      {viewerPage !== null && (
        <PageViewer
          page={viewerPage}
          onNavigate={openViewer}
          onClose={closeViewer}
        />
      )}
    </div>
  );
}

function StatusPill({ status }: { status: BackendStatus }) {
  const config = {
    connecting: {
      dot: "animate-pulse bg-highlight-foreground",
      label: "Waking backend…",
      title: "Pinging the hosted backend — cold starts can take up to a minute",
    },
    online: {
      dot: "bg-chart-2",
      label: "Backend online",
      title: "The hosted backend is responding",
    },
    offline: {
      dot: "bg-destructive",
      label: "Backend unreachable",
      title: "Could not reach the hosted backend",
    },
  }[status];

  return (
    <span
      title={config.title}
      className="flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1.5 text-[11px] text-muted-foreground"
    >
      <span className={`size-1.5 rounded-full ${config.dot}`} />
      <span className="hidden md:inline">{config.label}</span>
    </span>
  );
}

function EmptyState({ onAsk }: { onAsk: (question: string) => void }) {
  return (
    <div className="animate-fade-up flex flex-col items-center pt-6 text-center sm:pt-12">
      <div className="flex size-12 items-center justify-center rounded-full border border-border bg-card">
        <FlaskConical className="size-5 text-muted-foreground" />
      </div>
      <h2 className="mt-5 max-w-md font-serif text-2xl font-semibold leading-snug tracking-tight text-foreground sm:text-[1.7rem]">
        Ask questions about{" "}
        <span className="italic">“Attention Is All You Need”</span>.
      </h2>
      <p className="mt-3 max-w-md text-sm leading-6 text-muted-foreground">
        The system adaptively decides whether visual reasoning is required, and
        answers with cited pages from the paper.
      </p>

      <div className="mt-8 grid w-full gap-2 sm:grid-cols-2">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onAsk(q)}
            className="rounded-xl border border-border bg-card px-4 py-3 text-left text-sm leading-6 text-foreground transition-colors hover:border-ring/50 hover:bg-accent"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
