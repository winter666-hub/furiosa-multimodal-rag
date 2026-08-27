import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Loader2,
  Minus,
  Plus,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";
import type { AskSource } from "@/lib/ask-types";
import {
  MAX_ZOOM,
  MIN_ZOOM,
  ZOOM_STEP,
  clampZoom,
  fitPageSize,
  zoomPageSize,
  type PageSize,
} from "./page-viewer-zoom";

export function PageViewer({
  documentId,
  filename,
  page,
  totalPages,
  source,
  onNavigate,
  onClose,
  onDocumentLost,
}: {
  documentId: string;
  filename: string;
  page: number;
  totalPages: number;
  source?: AskSource;
  onNavigate: (page: number) => void;
  onClose: () => void;
  onDocumentLost: () => void;
}) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [attempt, setAttempt] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [basePageSize, setBasePageSize] = useState<PageSize>();
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageUrl = `/api/public/documents/${encodeURIComponent(documentId)}/pages/${page}`;

  const updateBasePageSize = useCallback(() => {
    const container = scrollContainerRef.current;
    const image = imageRef.current;
    if (!container || !image?.naturalWidth || !image.naturalHeight) return;

    const styles = window.getComputedStyle(container);
    const horizontalPadding = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
    const verticalPadding = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
    setBasePageSize(
      fitPageSize(
        image.naturalWidth,
        image.naturalHeight,
        container.clientWidth - horizontalPadding,
        Math.min(window.innerHeight * 0.66, container.clientHeight - verticalPadding),
      ),
    );
  }, []);

  useEffect(() => {
    setStatus("loading");
    setBasePageSize(undefined);
  }, [page, documentId, attempt]);
  useEffect(() => {
    setZoom(1);
    scrollContainerRef.current?.scrollTo({ top: 0, left: 0 });
  }, [page, documentId, source?.chunk_id]);
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const isEditable =
        target?.isContentEditable ||
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.tagName === "SELECT";
      if (isEditable) return;

      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft" && page > 1) onNavigate(page - 1);
      if (event.key === "ArrowRight" && page < totalPages) onNavigate(page + 1);
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        setZoom((value) => clampZoom(value + ZOOM_STEP));
      }
      if (event.key === "-") {
        event.preventDefault();
        setZoom((value) => clampZoom(value - ZOOM_STEP));
      }
      if (event.key === "0") {
        event.preventDefault();
        setZoom(1);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [page, totalPages, onNavigate, onClose]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    function onWheel(event: WheelEvent) {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      setZoom((value) => clampZoom(value + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP)));
    }

    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, []);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(updateBasePageSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [updateBasePageSize]);

  const zoomedPageSize = basePageSize ? zoomPageSize(basePageSize, zoom) : undefined;

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
              {source ? "Source · " : ""}Page {page} / {totalPages}
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
        <div className="flex flex-wrap items-center justify-center gap-1.5 border-b border-border bg-card px-3 py-2">
          <button
            type="button"
            onClick={() => setZoom((value) => clampZoom(value - ZOOM_STEP))}
            disabled={zoom <= MIN_ZOOM}
            aria-label="Zoom out"
            title="Zoom out (-)"
            className="inline-flex size-8 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Minus className="size-3.5" />
          </button>
          <output
            aria-live="polite"
            aria-label="Current zoom"
            className="min-w-14 text-center font-mono text-xs tabular-nums text-foreground"
          >
            {Math.round(zoom * 100)}%
          </output>
          <button
            type="button"
            onClick={() => setZoom((value) => clampZoom(value + ZOOM_STEP))}
            disabled={zoom >= MAX_ZOOM}
            aria-label="Zoom in"
            title="Zoom in (+)"
            className="inline-flex size-8 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Plus className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setZoom(1)}
            disabled={zoom === 1}
            title="Reset zoom (0)"
            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border px-2.5 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          >
            <RotateCcw className="size-3.5" />
            Reset
          </button>
        </div>
        <div
          ref={scrollContainerRef}
          className="relative flex min-h-72 min-w-0 flex-1 overflow-auto overscroll-contain bg-muted/50 p-3 sm:p-4"
        >
          {status === "loading" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
              <Loader2 className="size-5 animate-spin" />
              <p className="text-xs">페이지를 불러오는 중입니다</p>
            </div>
          )}
          {status === "error" ? (
            <div className="m-auto text-center">
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
            <div
              data-testid="zoomed-page-canvas"
              className="relative m-auto shrink-0"
              style={zoomedPageSize}
            >
              <img
                ref={imageRef}
                key={`${documentId}-${page}-${attempt}`}
                src={`${imageUrl}?attempt=${attempt}`}
                alt={`${filename}의 ${page}페이지`}
                onLoad={() => {
                  updateBasePageSize();
                  setStatus("ready");
                }}
                onError={() => setStatus("error")}
                className={`block rounded-sm border border-border bg-card shadow-md transition-opacity ${basePageSize ? "size-full max-h-none max-w-none" : "h-auto max-h-[66dvh] w-auto max-w-full"} ${status === "ready" ? "opacity-100" : "opacity-0"}`}
              />
              {status === "ready" && source?.page_width && source.page_height
                ? source.highlights.map((highlight, index) => (
                    <span
                      key={`${highlight.x}-${highlight.y}-${index}`}
                      aria-hidden="true"
                      className="pointer-events-none absolute rounded-sm border border-highlight-foreground/40 bg-highlight/45"
                      style={{
                        left: `${(highlight.x / source.page_width) * 100}%`,
                        top: `${(highlight.y / source.page_height) * 100}%`,
                        width: `${(highlight.width / source.page_width) * 100}%`,
                        height: `${(highlight.height / source.page_height) * 100}%`,
                      }}
                    />
                  ))
                : null}
            </div>
          )}
        </div>
        {source?.excerpt && (
          <section className="border-t border-border bg-card px-4 py-3">
            <h2 className="text-xs font-semibold text-foreground">Referenced passage</h2>
            <blockquote className="mt-1.5 max-h-28 overflow-y-auto border-l-2 border-highlight-foreground/40 pl-3 text-xs leading-5 text-muted-foreground">
              “{source.excerpt}”
            </blockquote>
          </section>
        )}
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
