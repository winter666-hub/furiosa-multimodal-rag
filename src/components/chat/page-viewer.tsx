import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Loader2, X } from "lucide-react";

// Matches MAX_DOCUMENT_PAGE in the document-page proxy route.
export const MAX_DOCUMENT_PAGE = 15;

export function PageViewer({
  page,
  onNavigate,
  onClose,
}: {
  page: number;
  onNavigate: (page: number) => void;
  onClose: () => void;
}) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );

  useEffect(() => {
    setStatus("loading");
  }, [page]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && page > 1) onNavigate(page - 1);
      if (e.key === "ArrowRight" && page < MAX_DOCUMENT_PAGE)
        onNavigate(page + 1);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [page, onNavigate, onClose]);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Document page ${page}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/45 p-3 backdrop-blur-[2px] sm:p-6"
      onClick={onClose}
    >
      <div
        className="animate-fade-up flex max-h-full w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-[0_16px_48px_oklch(0.2_0.02_262/0.35)]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
          <p className="font-mono text-xs tracking-wide text-muted-foreground">
            Attention Is All You Need ·{" "}
            <span className="text-foreground">
              Page {page} / {MAX_DOCUMENT_PAGE}
            </span>
          </p>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close page viewer"
            className="flex size-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="relative flex min-h-64 flex-1 items-center justify-center overflow-y-auto bg-muted/50 p-3 sm:p-4">
          {status === "loading" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
              <Loader2 className="size-5 animate-spin" />
              <p className="text-xs">Loading page…</p>
            </div>
          )}
          {status === "error" ? (
            <p className="text-sm text-muted-foreground">
              This page couldn't be loaded. The backend may be waking up — try
              again in a moment.
            </p>
          ) : (
            <img
              key={page}
              src={`/api/public/document-page/${page}`}
              alt={`Page ${page} of the paper "Attention Is All You Need"`}
              onLoad={() => setStatus("ready")}
              onError={() => setStatus("error")}
              className={`max-h-[70dvh] w-auto max-w-full rounded-sm border border-border bg-card shadow-[0_2px_10px_oklch(0.3_0.02_262/0.12)] transition-opacity ${
                status === "ready" ? "opacity-100" : "opacity-0"
              }`}
            />
          )}
        </div>

        <div className="flex items-center justify-between border-t border-border px-3 py-2">
          <button
            type="button"
            onClick={() => onNavigate(page - 1)}
            disabled={page <= 1}
            className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2.5 py-1 text-xs font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            <ChevronLeft className="size-3.5" />
            Prev
          </button>
          <p className="text-[11px] text-muted-foreground">
            Use ← → keys to navigate
          </p>
          <button
            type="button"
            onClick={() => onNavigate(page + 1)}
            disabled={page >= MAX_DOCUMENT_PAGE}
            className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2.5 py-1 text-xs font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
            <ChevronRight className="size-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
