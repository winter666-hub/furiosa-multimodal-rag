import { useEffect, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Loader2,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";

export function PageViewer({
  documentId,
  filename,
  page,
  totalPages,
  onNavigate,
  onClose,
  onDocumentLost,
}: {
  documentId: string;
  filename: string;
  page: number;
  totalPages: number;
  onNavigate: (page: number) => void;
  onClose: () => void;
  onDocumentLost: () => void;
}) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [attempt, setAttempt] = useState(0);
  const imageUrl = `/api/public/documents/${encodeURIComponent(documentId)}/pages/${page}`;

  useEffect(() => setStatus("loading"), [page, documentId, attempt]);
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft" && page > 1) onNavigate(page - 1);
      if (event.key === "ArrowRight" && page < totalPages) onNavigate(page + 1);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [page, totalPages, onNavigate, onClose]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`문서 ${page}페이지`}
      className="fixed inset-0 z-50 flex items-end justify-center bg-foreground/45 p-0 backdrop-blur-[2px] sm:items-center sm:p-6"
      onClick={onClose}
    >
      <div
        className="animate-fade-up flex max-h-[94dvh] w-full max-w-3xl flex-col overflow-hidden rounded-t-2xl border border-border bg-card shadow-2xl sm:max-h-full sm:rounded-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <p className="truncate text-xs text-muted-foreground">{filename}</p>
            <p className="font-mono text-xs">
              Page {page} / {totalPages}
            </p>
          </div>
          <div className="flex items-center gap-1">
            <a
              href={imageUrl}
              target="_blank"
              rel="noreferrer"
              aria-label="새 탭에서 열기"
              className="flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <ExternalLink className="size-4" />
            </a>
            <button
              type="button"
              onClick={onClose}
              aria-label="페이지 뷰어 닫기"
              className="flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <X className="size-4" />
            </button>
          </div>
        </header>
        <div className="relative flex min-h-72 flex-1 items-center justify-center overflow-auto bg-muted/50 p-3 sm:p-4">
          {status === "loading" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
              <Loader2 className="size-5 animate-spin" />
              <p className="text-xs">페이지를 불러오는 중입니다</p>
            </div>
          )}
          {status === "error" ? (
            <div className="text-center">
              <p className="text-sm text-muted-foreground">
                페이지를 불러오지 못했습니다. 서버에서 문서가 만료됐을 수 있습니다.
              </p>
              <div className="mt-3 flex justify-center gap-2">
                <button
                  type="button"
                  onClick={() => setAttempt((value) => value + 1)}
                  className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium hover:bg-accent"
                >
                  <RotateCcw className="size-3.5" />
                  다시 시도
                </button>
                <button
                  type="button"
                  onClick={onDocumentLost}
                  className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground"
                >
                  <Upload className="size-3.5" />
                  PDF 다시 업로드
                </button>
              </div>
            </div>
          ) : (
            <img
              key={`${documentId}-${page}-${attempt}`}
              src={`${imageUrl}?attempt=${attempt}`}
              alt={`${filename}의 ${page}페이지`}
              onLoad={() => setStatus("ready")}
              onError={() => setStatus("error")}
              className={`max-h-[72dvh] w-auto max-w-full rounded-sm border border-border bg-card shadow-md transition-opacity ${status === "ready" ? "opacity-100" : "opacity-0"}`}
            />
          )}
        </div>
        <footer className="flex items-center justify-between border-t border-border px-3 py-2">
          <button
            type="button"
            onClick={() => onNavigate(page - 1)}
            disabled={page <= 1}
            className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs font-medium hover:bg-accent disabled:opacity-40"
          >
            <ChevronLeft className="size-3.5" />
            이전
          </button>
          <span className="text-[11px] text-muted-foreground">방향키로 이동</span>
          <button
            type="button"
            onClick={() => onNavigate(page + 1)}
            disabled={page >= totalPages}
            className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs font-medium hover:bg-accent disabled:opacity-40"
          >
            다음
            <ChevronRight className="size-3.5" />
          </button>
        </footer>
      </div>
    </div>
  );
}
