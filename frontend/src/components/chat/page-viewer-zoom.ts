export const MIN_ZOOM = 0.75;
export const MAX_ZOOM = 3;
export const ZOOM_STEP = 0.25;

export function clampZoom(value: number) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));
}
