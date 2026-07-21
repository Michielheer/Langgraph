// 3D-visualisatie "dekking vs normprofiel". Staven = huidige dekking per
// categorie; coral kap = het tekort tot de norm (onderverzekering/hiaat).
// Morpht vloeiend tussen datasets (particulier <-> zakelijk).
import {
  AmbientLight,
  BoxGeometry,
  CanvasTexture,
  DirectionalLight,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  WebGLRenderer,
} from "three";

const MINT = 0xa6f2d2;
const CORAL = 0xf25a5a;
const PERIWINKLE = 0x576cdb;
const BASECOL = 0xe6e9f4;

const SPACING = 1.15;
const MAXH = 3.2;
const BARW = 0.62;

export interface BarSpec {
  code: string;
  actual: number; // 0..1
  norm: number; // 0..1
}
export interface Track {
  key: string;
  bars: BarSpec[];
}
export interface CoverageOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  width: number;
  height: number;
  dpr: number;
  tracks: Track[];
  initial: string;
  reducedMotion: boolean;
}
export interface CoverageApi {
  resize: (width: number, height: number) => void;
  setPointer: (x: number, y: number) => void;
  clearPointer: () => void;
  setTrack: (key: string) => void;
}

const makeLabelTexture = (label: string) => {
  const c =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(128, 128)
      : (Object.assign(document.createElement("canvas"), { width: 128, height: 128 }) as HTMLCanvasElement);
  const ctx = c.getContext("2d") as CanvasRenderingContext2D;
  ctx.fillStyle = "#ffffff";
  ctx.font = "600 30px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, 64, 66);
  return new CanvasTexture(c as HTMLCanvasElement);
};

export function initCoverageScene(opts: CoverageOptions): CoverageApi {
  const { canvas, dpr, tracks, initial, reducedMotion } = opts;
  const trackMap = new Map(tracks.map((t) => [t.key, t]));
  const n = tracks[0].bars.length;

  const renderer = new WebGLRenderer({
    canvas: canvas as HTMLCanvasElement,
    alpha: true,
    antialias: true,
    powerPreference: "low-power",
  });
  renderer.setPixelRatio(dpr);

  const scene = new Scene();
  const camera = new PerspectiveCamera(36, opts.width / opts.height, 0.1, 100);
  camera.position.set(5.2, 5.4, 7.6);
  camera.lookAt(0, 1.15, 0);

  scene.add(new AmbientLight(0xffffff, 0.92));
  const dir = new DirectionalLight(0xffffff, 0.6);
  dir.position.set(5, 11, 7);
  scene.add(dir);

  const group = new Group();
  scene.add(group);

  const offX = ((n - 1) * SPACING) / 2;

  // Basisplateau
  const base = new Mesh(
    new BoxGeometry(n * SPACING + 0.7, 0.2, 2),
    new MeshLambertMaterial({ color: BASECOL }),
  );
  base.position.y = -0.1;
  group.add(base);

  // Label-textures voor alle codes (particulier + zakelijk)
  const labelTextures = new Map<string, CanvasTexture>();
  for (const t of tracks) for (const b of t.bars) {
    if (!labelTextures.has(b.code)) labelTextures.set(b.code, makeLabelTexture(b.code));
  }

  const barGeo = new BoxGeometry(BARW, 1, BARW);
  const labelGeo = new PlaneGeometry(BARW * 1.35, BARW * 1.35);

  interface Bar {
    covered: Mesh;
    coveredMat: MeshLambertMaterial;
    gap: Mesh;
    gapMat: MeshBasicMaterial;
    normTick: Mesh;
    labelMat: MeshBasicMaterial;
    curActual: number;
    curNorm: number;
    tActual: number;
    tNorm: number;
  }
  const bars: Bar[] = [];

  for (let i = 0; i < n; i++) {
    const x = i * SPACING - offX;

    const coveredMat = new MeshLambertMaterial({ color: MINT, emissive: MINT, emissiveIntensity: 0.08 });
    const covered = new Mesh(barGeo, coveredMat);
    covered.position.x = x;
    group.add(covered);

    const gapMat = new MeshBasicMaterial({ color: CORAL, transparent: true, opacity: 0.42, depthWrite: false });
    const gap = new Mesh(barGeo, gapMat);
    gap.position.x = x;
    group.add(gap);

    const normTick = new Mesh(
      new BoxGeometry(BARW * 1.2, 0.035, BARW * 1.2),
      new MeshBasicMaterial({ color: PERIWINKLE, transparent: true, opacity: 0.9 }),
    );
    normTick.position.x = x;
    group.add(normTick);

    const labelMat = new MeshBasicMaterial({ transparent: true, color: 0x575775 });
    const label = new Mesh(labelGeo, labelMat);
    label.rotation.x = -Math.PI / 2;
    label.position.set(x, 0.02, 0.92);
    group.add(label);

    bars.push({
      covered,
      coveredMat,
      gap,
      gapMat,
      normTick,
      labelMat,
      curActual: 0,
      curNorm: 0,
      tActual: 0,
      tNorm: 0,
    });
  }

  const applyTrack = (key: string) => {
    const track = trackMap.get(key) ?? tracks[0];
    track.bars.forEach((spec, i) => {
      const bar = bars[i];
      bar.tActual = spec.actual;
      bar.tNorm = spec.norm;
      const tex = labelTextures.get(spec.code)!;
      bar.labelMat.map = tex;
      bar.labelMat.needsUpdate = true;
    });
  };
  applyTrack(initial);

  const updateBar = (bar: Bar) => {
    const aH = Math.max(bar.curActual, 0.001) * MAXH;
    bar.covered.scale.y = aH;
    bar.covered.position.y = aH / 2;

    const gapH = Math.max(bar.curNorm - bar.curActual, 0) * MAXH;
    const show = gapH > 0.04;
    bar.gap.visible = show;
    if (show) {
      bar.gap.scale.y = gapH;
      bar.gap.position.y = aH + gapH / 2;
    }
    bar.normTick.position.y = Math.max(bar.curNorm, 0.001) * MAXH;

    // Kleur: grote tekorten laten de staaf richting coral neigen
    const deficit = Math.max(bar.curNorm - bar.curActual, 0);
    const warn = Math.min(deficit / 0.4, 1);
    bar.coveredMat.emissiveIntensity = 0.06 + warn * 0.12;
  };

  const setSize = (w: number, h: number) => {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  setSize(opts.width, opts.height);

  let targetRotY = 0;
  let targetRotX = 0;

  const api: CoverageApi = {
    resize: (w, h) => {
      setSize(w, h);
      if (reducedMotion) renderer.render(scene, camera);
    },
    setPointer: (x, y) => {
      targetRotY = x * 0.3;
      targetRotX = y * 0.14;
    },
    clearPointer: () => {
      targetRotY = 0;
      targetRotX = 0;
    },
    setTrack: (key) => {
      applyTrack(key);
      if (reducedMotion) {
        bars.forEach((b) => {
          b.curActual = b.tActual;
          b.curNorm = b.tNorm;
          updateBar(b);
        });
        renderer.render(scene, camera);
      }
    },
  };

  if (reducedMotion) {
    bars.forEach((b) => {
      b.curActual = b.tActual;
      b.curNorm = b.tNorm;
      updateBar(b);
    });
    renderer.render(scene, camera);
    return api;
  }

  const animate = (now: number) => {
    requestAnimationFrame(animate);
    for (const bar of bars) {
      bar.curActual += (bar.tActual - bar.curActual) * 0.09;
      bar.curNorm += (bar.tNorm - bar.curNorm) * 0.09;
      updateBar(bar);
    }
    const idle = Math.sin(now / 5600) * 0.05;
    group.rotation.y += (idle + targetRotY - group.rotation.y) * 0.06;
    group.rotation.x += (targetRotX - group.rotation.x) * 0.06;
    renderer.render(scene, camera);
  };
  requestAnimationFrame(animate);

  return api;
}
