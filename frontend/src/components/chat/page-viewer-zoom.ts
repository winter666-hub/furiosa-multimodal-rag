export const MIN_ZOOM = 0.75;
export const MAX_ZOOM = 3;
export const ZOOM_STEP = 0.25;

export interface PageSize {
  width: number;
  height: number;
}

export function clampZoom(value: number) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));
}

export function fitPageSize(
  naturalWidth: number,
  naturalHeight: number,
  availableWidth: number,
  availableHeight: number,
): PageSize {
  const fitScale = Math.min(1, availableWidth / naturalWidth, availableHeight / naturalHeight);
  return { width: naturalWidth * fitScale, height: naturalHeight * fitScale };
}

export function zoomPageSize(baseSize: PageSize, zoom: number): PageSize {
  return { width: baseSize.width * zoom, height: baseSize.height * zoom };
}
