import { describe, expect, test } from "bun:test";
import {
  MAX_ZOOM,
  MIN_ZOOM,
  ZOOM_STEP,
  clampZoom,
  fitPageSize,
  zoomPageSize,
} from "./page-viewer-zoom";

describe("source page zoom", () => {
  test("advances from 100% to 300% in 25% steps", () => {
    const levels = [1];
    while (levels.at(-1)! < MAX_ZOOM) {
      levels.push(clampZoom(levels.at(-1)! + ZOOM_STEP));
    }

    expect(levels).toEqual([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]);
    expect(clampZoom(MAX_ZOOM + ZOOM_STEP)).toBe(MAX_ZOOM);
  });

  test("does not zoom out below 75%", () => {
    expect(clampZoom(MIN_ZOOM - ZOOM_STEP)).toBe(MIN_ZOOM);
  });

  test("scales the actual page canvas beyond the fitted viewport size", () => {
    const baseSize = fitPageSize(1785, 2526, 600, 700);
    const at100 = zoomPageSize(baseSize, 1);
    const at150 = zoomPageSize(baseSize, 1.5);
    const at200 = zoomPageSize(baseSize, 2);
    const at300 = zoomPageSize(baseSize, 3);

    expect(at150.width).toBe(at100.width * 1.5);
    expect(at200.width).toBe(at100.width * 2);
    expect(at300.width).toBe(at100.width * 3);
    expect(at300.width).toBeGreaterThan(600);
    expect(at300.height).toBeGreaterThan(700);
  });
});
