import { useEffect, useRef, useState } from "react";
import { FileText, Loader2, RefreshCw, UploadCloud } from "lucide-react";
import type { CurrentDocument } from "@/lib/ask-types";

const MAX_FILE_SIZE = 25 * 1024 * 1024;

type UploadPayload = {
  document_id: string;
  filename: string;
  pages: number;
  status: "ready";
  cache_hit?: boolean;
};

export function DocumentUpload({
  document,
  onUploaded,
  onReplace,
}: {
  document: CurrentDocument | null;
  onUploaded: (document: CurrentDocument) => void;
  onReplace: () => boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [coldStart, setColdStart] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!uploading) return;
    const timer = window.setTimeout(() => setColdStart(true), 20_000);
    return () => window.clearTimeout(timer);
  }, [uploading]);

  async function upload(file: File) {
    setError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("PDF 파일만 업로드할 수 있습니다.");
      return;
    }
    if (file.type && file.type !== "application/pdf") {
      setError("선택한 파일의 형식이 PDF가 아닙니다.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("PDF 크기는 25MB 이하여야 합니다.");
      return;
    }
    if (document && !onReplace()) return;

    setUploading(true);
    setColdStart(false);
    try {
      const formData = new FormData();
      formData.set("file", file);
      const response = await fetch("/api/public/documents", { method: "POST", body: formData });
      const payload = (await response.json().catch(() => null)) as
        UploadPayload | { detail?: string } | null;
      if (!response.ok || !payload || !("document_id" in payload)) {
        throw new Error(
          payload && "detail" in payload && payload.detail
            ? payload.detail
            : "PDF 업로드에 실패했습니다.",
        );
      }
      onUploaded({
        documentId: payload.document_id,
        filename: payload.filename,
        pages: payload.pages,
        status: payload.status,
        ...(payload.cache_hit === undefined ? {} : { cacheHit: payload.cache_hit }),
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "PDF 업로드에 실패했습니다.");
    } finally {
      setUploading(false);
      setColdStart(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  const fileInput = (
    <input
      ref={inputRef}
      hidden
      type="file"
      accept="application/pdf,.pdf"
      onChange={(event) => {
        const file = event.target.files?.[0];
        if (file) void upload(file);
      }}
    />
  );

  if (document) {
    return (
      <section className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-secondary">
            <FileText className="size-5" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold">{document.filename}</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {document.pages}페이지 · 분석 준비 완료{document.cacheHit ? " · 캐시 사용" : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium hover:bg-accent disabled:opacity-50"
          >
            <RefreshCw className="size-3.5" />
            교체
          </button>
        </div>
        {uploading && (
          <p className="mt-3 text-xs text-muted-foreground">
            새 PDF를 업로드하고 준비하는 중입니다…
          </p>
        )}
        {error && <p className="mt-3 text-xs text-destructive">{error}</p>}
        {fileInput}
      </section>
    );
  }

  return (
    <section>
      <button
        type="button"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files[0];
          if (file) void upload(file);
        }}
        className={`flex w-full flex-col items-center rounded-2xl border border-dashed px-6 py-8 text-center transition-colors ${dragging ? "border-ring bg-accent" : "border-input bg-card hover:border-ring/60 hover:bg-accent/50"}`}
      >
        {uploading ? (
          <Loader2 className="size-7 animate-spin text-muted-foreground" />
        ) : (
          <UploadCloud className="size-7 text-muted-foreground" />
        )}
        <span className="mt-3 text-sm font-semibold">
          {uploading
            ? "PDF를 업로드하고 문서를 준비하는 중입니다"
            : "PDF를 놓거나 클릭해 업로드하세요"}
        </span>
        <span className="mt-1 text-xs text-muted-foreground">
          {coldStart
            ? "호스팅 서버가 시작 중일 수 있습니다. 잠시만 기다려 주세요."
            : "PDF · 최대 25MB"}
        </span>
      </button>
      {error && <p className="mt-2 text-center text-xs text-destructive">{error}</p>}
      {fileInput}
    </section>
  );
}
