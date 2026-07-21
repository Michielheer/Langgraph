// Web Worker die de dekking-vs-norm scène op een OffscreenCanvas draait.
import { initCoverageScene, type CoverageApi } from "./coverageScene";

let api: CoverageApi | undefined;

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "init") {
    api = initCoverageScene({
      canvas: m.canvas,
      width: m.width,
      height: m.height,
      dpr: m.dpr,
      tracks: m.tracks,
      initial: m.initial,
      reducedMotion: m.reducedMotion,
    });
  } else if (api) {
    if (m.type === "resize") api.resize(m.width, m.height);
    else if (m.type === "pointer") api.setPointer(m.x, m.y);
    else if (m.type === "pointerleave") api.clearPointer();
    else if (m.type === "track") api.setTrack(m.key);
  }
};
