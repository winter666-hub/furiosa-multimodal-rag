import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpenText, FileSearch } from "lucide-react";
import { askPaperQuestion, getBackendHealth } from "@/lib/ask.functions";
import type { AskSource, CurrentDocument } from "@/lib/ask-types";
import { ChatEntryView, type ChatEntry } from "@/components/chat/chat-message";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { PageViewer } from "@/components/chat/page-viewer";
import { DocumentUpload } from "@/components/document-upload";
import { getSessionId } from "@/lib/session";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Furiosa Paper RAG · Selective Multimodal Research Demo" },
      {
        name: "description",
        content:
          "Upload a research paper and ask grounded questions with adaptive text and visual routing.",
      },
      { property: "og:title", content: "Furiosa Paper RAG" },
      {
        property: "og:description",
        content: "Selective multimodal question answering for your PDF.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: DemoPage,
});

const SUGGESTED_QUESTIONS = [
  "이 논문의 핵심 기여를 세 가지로 요약해줘.",
  "제안한 방법의 전체 구조와 각 구성 요소의 역할은 무엇인가?",
  "실험 결과에서 기존 방법과 비교해 가장 크게 개선된 지표는 무엇인가?",
  "저자가 언급한 한계와 향후 연구 방향을 설명해줘.",
];

type BackendStatus = "connecting" | "online" | "offline";
type ViewerState = {
  documentId: string;
  filename: string;
  page: number;
  totalPages: number;
  source?: AskSource;
};

function DemoPage() {
  const [document, setDocument] = useState<CurrentDocument | null>(null);
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("connecting");
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [entries, pending]);

  useEffect(() => {
    let cancelled = false;
    void getBackendHealth().then((result) => {
      if (!cancelled) setBackendStatus(result.ok ? "online" : "offline");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const closeViewer = useCallback(() => setViewer(null), []);
  const navigateViewer = useCallback((page: number) => {
    setViewer((current) =>
      current
        ? { ...current, page, source: current.source?.page === page ? current.source : undefined }
        : null,
    );
  }, []);
  const openViewer = useCallback(
    (source: AskSource, documentId: string) => {
      const current = document;
      if (!current || current.documentId !== documentId) return;
      setViewer({
        documentId,
        filename: current.filename,
        page: source.page,
        totalPages: current.pages,
        source,
      });
    },
    [document],
  );

  async function send(rawQuestion: string) {
    const question = rawQuestion.trim();
    const currentDocument = document;
    if (!question || pending || !currentDocument) return;

    setInput("");
    setPending(true);
    setEntries((previous) => [
      ...previous,
      { id: crypto.randomUUID(), role: "user", text: question },
    ]);
    const result = await askPaperQuestion({
      data: {
        question,
        documentId: currentDocument.documentId,
        sessionId: getSessionId(),
      },
    });
    setEntries((previous) => [
      ...previous,
      result.ok
        ? { id: crypto.randomUUID(), role: "assistant", data: result.data }
        : { id: crypto.randomUUID(), role: "error", status: result.status, question },
    ]);
    if (result.ok) setBackendStatus("online");
    setPending(false);
    inputRef.current?.focus();
  }

  function confirmReplacement() {
    if (
      entries.length > 0 &&
      !window.confirm("PDF를 교체하면 현재 대화 기록이 삭제됩니다. 계속할까요?")
    )
      return false;
    setEntries([]);
    setViewer(null);
    return true;
  }

  function resetMissingDocument() {
    setDocument(null);
    setViewer(null);
    setEntries([]);
  }

  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="border-b border-border bg-background">
        <div className="mx-auto flex w-full max-w-4xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <div className="min-w-0">
            <h1 className="truncate font-serif text-lg font-semibold tracking-tight">
              Furiosa Paper RAG
            </h1>
            <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Selective Multimodal Research Demo
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <StatusPill status={backendStatus} />
            {document && (
              <button
                type="button"
                onClick={() =>
                  setViewer({
                    documentId: document.documentId,
                    filename: document.filename,
                    page: 1,
                    totalPages: document.pages,
                  })
                }
                title="문서 첫 페이지 보기"
                className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                <BookOpenText className="size-3.5" />
                <span className="hidden max-w-48 truncate sm:inline">{document.filename}</span>
              </button>
            )}
          </div>
        </div>
      </header>

      <main ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-4xl flex-col gap-5 px-4 py-7 sm:px-6">
          <DocumentUpload
            document={document}
            onReplace={confirmReplacement}
            onUploaded={(next) => {
              setDocument(next);
              setEntries([]);
              setBackendStatus("online");
            }}
          />
          {!document ? (
            <Welcome />
          ) : entries.length === 0 && !pending ? (
            <EmptyState onAsk={(question) => void send(question)} />
          ) : (
            entries.map((entry) => (
              <ChatEntryView
                key={entry.id}
                entry={entry}
                onRetry={(question) => void send(question)}
                onViewPage={openViewer}
                onDocumentLost={resetMissingDocument}
              />
            ))
          )}
          {pending && <TypingIndicator />}
        </div>
      </main>

      <footer className="border-t border-border bg-background">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void send(input);
          }}
          className="mx-auto w-full max-w-4xl px-4 py-4 sm:px-6"
        >
          <div className="flex items-end gap-2 rounded-2xl border border-input bg-card p-2 shadow-sm focus-within:border-ring/60">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  void send(input);
                }
              }}
              disabled={!document || pending}
              placeholder={
                document ? "업로드한 논문에 대해 질문하세요…" : "먼저 PDF를 업로드해 주세요"
              }
              rows={1}
              className="field-sizing-content max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-6 outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!document || pending || !input.trim()}
              aria-label="질문 보내기"
              className="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-35"
            >
              <ArrowUp className="size-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            응답은 업로드한 PDF만 근거로 생성되며, 콜드 스타트 시 최대 1분 이상 걸릴 수 있습니다.
          </p>
          <p className="mt-1 text-center text-[11px] text-muted-foreground">
            Questions and AI responses may be stored for service improvement and research analysis.
            Do not submit sensitive or personal information.
          </p>
        </form>
      </footer>

      {viewer && (
        <PageViewer
          {...viewer}
          onNavigate={navigateViewer}
          onClose={closeViewer}
          onDocumentLost={resetMissingDocument}
        />
      )}
    </div>
  );
}

function StatusPill({ status }: { status: BackendStatus }) {
  const config = {
    connecting: ["animate-pulse bg-highlight-foreground", "서버 연결 중"],
    online: ["bg-chart-2", "서버 연결됨"],
    offline: ["bg-destructive", "서버 연결 안 됨"],
  }[status];
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1.5 text-[11px] text-muted-foreground">
      <span className={`size-1.5 rounded-full ${config[0]}`} />
      <span className="hidden md:inline">{config[1]}</span>
    </span>
  );
}

function Welcome() {
  return (
    <div className="animate-fade-up flex flex-col items-center py-5 text-center sm:py-10">
      <div className="flex size-12 items-center justify-center rounded-full border border-border bg-card">
        <FileSearch className="size-5 text-muted-foreground" />
      </div>
      <h2 className="mt-5 font-serif text-2xl font-semibold">
        내 논문에서 근거 있는 답을 찾아보세요
      </h2>
      <p className="mt-3 max-w-lg text-sm leading-6 text-muted-foreground">
        PDF를 업로드하면 질문의 성격에 따라 텍스트 검색과 시각 분석을 선택하고, 답변의 출처 페이지를
        함께 보여줍니다.
      </p>
    </div>
  );
}

function EmptyState({ onAsk }: { onAsk: (question: string) => void }) {
  return (
    <div className="animate-fade-up pt-2 text-center">
      <h2 className="font-serif text-xl font-semibold">무엇이 궁금한가요?</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        직접 질문하거나 아래 예시로 시작해 보세요.
      </p>
      <div className="mt-5 grid gap-2 sm:grid-cols-2">
        {SUGGESTED_QUESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            onClick={() => onAsk(question)}
            className="rounded-xl border border-border bg-card px-4 py-3 text-left text-sm leading-6 hover:border-ring/50 hover:bg-accent"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}
