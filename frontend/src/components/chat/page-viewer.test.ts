import { describe, expect, test } from "bun:test";
import { MAX_ZOOM, MIN_ZOOM, ZOOM_STEP, clampZoom } from "./page-viewer-zoom";

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
});
