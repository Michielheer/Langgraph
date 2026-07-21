// Web Worker die de 3D-scanscène op een OffscreenCanvas draait, zodat
// three.js-parse en rendering de main thread niet blokkeren.
import { initScanScene, type ScanSceneApi } from "./scanScene";

let api: ScanSceneApi | undefined;

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "init") {
    api = initScanScene({
      canvas: m.canvas,
      width: m.width,
      height: m.height,
      dpr: m.dpr,
      cols: m.cols,
      rows: m.rows,
      cellSpecs: m.cellSpecs,
      reducedMotion: m.reducedMotion,
      setCounters: (t) => self.postMessage({ type: "counters", t }),
    });
  } else if (api) {
    if (m.type === "resize") api.resize(m.width, m.height);
    else if (m.type === "pointer") api.setPointer(m.x, m.y);
    else if (m.type === "pointerleave") api.clearPointer();
    else if (m.type === "rescan") api.rescan();
  }
};
