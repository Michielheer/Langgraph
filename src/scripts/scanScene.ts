// 3D live-scan scène voor de hero. Omgevingsneutraal: draait in een Web Worker
// (OffscreenCanvas) of op de main thread. Alle DOM-koppelingen lopen via de API.
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
  PointLight,
  Scene,
  WebGLRenderer,
} from "three";

const SPACING = 1.18;
const SCAN_MS = 3000;
const AMBER = 0xffb020;
const MINT = 0x4fd1a8;
const BASE = 0x0f3540;

export interface CellSpec {
  leak: boolean;
  label: string;
}

export interface ScanSceneOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  width: number;
  height: number;
  dpr: number;
  cols: number;
  rows: number;
  cellSpecs: CellSpec[];
  reducedMotion: boolean;
  setCounters: (t: number) => void;
}

export interface ScanSceneApi {
  resize: (width: number, height: number) => void;
  setPointer: (x: number, y: number) => void; // genormaliseerd, -0.5..0.5
  clearPointer: () => void;
  rescan: () => void;
}

const makeLabelCanvas = (label: string) => {
  const c =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(128, 128)
      : (Object.assign(document.createElement("canvas"), { width: 128, height: 128 }) as HTMLCanvasElement);
  const ctx = c.getContext("2d") as CanvasRenderingContext2D;
  ctx.fillStyle = "#ffffff";
  ctx.font = "600 34px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, 64, 66);
  return c;
};

export function initScanScene(opts: ScanSceneOptions): ScanSceneApi {
  const { canvas, dpr, cols, rows, cellSpecs, reducedMotion, setCounters } = opts;

  const renderer = new WebGLRenderer({
    canvas: canvas as HTMLCanvasElement,
    alpha: true,
    antialias: true,
    powerPreference: "low-power",
  });
  renderer.setPixelRatio(dpr);

  const scene = new Scene();
  const camera = new PerspectiveCamera(38, opts.width / opts.height, 0.1, 100);
  camera.position.set(6.9, 7.7, 9.6);
  camera.lookAt(0, -0.4, 0);

  scene.add(new AmbientLight(0xbfd8d2, 0.55));
  const dir = new DirectionalLight(0xffffff, 1.1);
  dir.position.set(6, 12, 8);
  scene.add(dir);
  const scanLight = new PointLight(AMBER, 0, 14, 1.6);
  scanLight.position.y = 1.6;
  scene.add(scanLight);

  const group = new Group();
  scene.add(group);

  // Label-textures (één per productcode), wit getekend zodat materiaalkleur ze tint
  const labelMaterials = new Map<string, MeshBasicMaterial>();
  for (const label of new Set(cellSpecs.map((c) => c.label))) {
    labelMaterials.set(
      label,
      new MeshBasicMaterial({
        map: new CanvasTexture(makeLabelCanvas(label) as HTMLCanvasElement),
        transparent: true,
        color: 0x83a29f,
      }),
    );
  }

  const tileGeo = new BoxGeometry(1, 0.22, 1);
  const labelGeo = new PlaneGeometry(0.72, 0.72);

  interface Cell {
    tile: Mesh;
    mat: MeshLambertMaterial;
    labelMat: MeshBasicMaterial;
    leak: boolean;
    x: number;
    lift: number;
    scanned: boolean;
  }
  const cells: Cell[] = [];
  const offX = ((cols - 1) * SPACING) / 2;
  const offZ = ((rows - 1) * SPACING) / 2;

  cellSpecs.forEach((spec, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);

    const mat = new MeshLambertMaterial({ color: BASE });
    const tile = new Mesh(tileGeo, mat);
    tile.position.set(col * SPACING - offX, 0, row * SPACING - offZ);
    group.add(tile);

    const labelMat = labelMaterials.get(spec.label)!.clone();
    const labelMesh = new Mesh(labelGeo, labelMat);
    labelMesh.rotation.x = -Math.PI / 2;
    labelMesh.position.set(0, 0.115, 0);
    tile.add(labelMesh);

    cells.push({ tile, mat, labelMat, leak: spec.leak, x: tile.position.x, lift: 0, scanned: false });
  });

  // Scanbalk: glowende amber wand die over de x-as veegt
  const barMat = new MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.55 });
  const bar = new Mesh(new BoxGeometry(0.06, 1.7, rows * SPACING + 1.2), barMat);
  bar.position.y = 0.85;
  group.add(bar);

  const setCellScanned = (cell: Cell) => {
    cell.scanned = true;
    if (cell.leak) {
      cell.mat.color.setHex(0x3d2c08);
      cell.mat.emissive.setHex(AMBER);
      cell.mat.emissiveIntensity = 0.75;
      cell.labelMat.color.setHex(0x241800);
    } else {
      cell.mat.color.setHex(0x123d3b);
      cell.mat.emissive.setHex(MINT);
      cell.mat.emissiveIntensity = 0.05;
      cell.labelMat.color.setHex(MINT);
    }
  };
  const resetCell = (cell: Cell) => {
    cell.scanned = false;
    cell.lift = 0;
    cell.mat.color.setHex(BASE);
    cell.mat.emissive.setHex(0x000000);
    cell.mat.emissiveIntensity = 0;
    cell.labelMat.color.setHex(0x83a29f);
    cell.tile.position.y = 0;
    cell.tile.scale.y = 1;
  };

  let targetRotY = 0;
  let targetRotX = 0;
  const startX = -offX - 1.2;
  const endX = offX + 1.2;
  let scanStart = performance.now();
  let scanDone = false;

  const finishInstant = () => {
    cells.forEach((c) => {
      setCellScanned(c);
      c.lift = c.leak ? 1 : 0;
      c.tile.position.y = c.leak ? 0.32 : 0;
      c.tile.scale.y = c.leak ? 2.6 : 1;
    });
    bar.visible = false;
    scanLight.intensity = 0;
    setCounters(1);
    group.rotation.set(0, 0, 0);
    renderer.render(scene, camera);
  };

  const setSize = (w: number, h: number) => {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  setSize(opts.width, opts.height);

  if (reducedMotion) {
    finishInstant();
    return {
      resize: (w, h) => {
        setSize(w, h);
        renderer.render(scene, camera);
      },
      setPointer: () => {},
      clearPointer: () => {},
      rescan: finishInstant,
    };
  }

  const animate = (now: number) => {
    requestAnimationFrame(animate);
    const t = Math.min((now - scanStart) / SCAN_MS, 1);
    const eased = 1 - Math.pow(1 - t, 2);

    const barX = startX + (endX - startX) * eased;
    bar.position.x = barX;
    scanLight.position.x = barX;
    scanLight.intensity = t < 1 ? 14 : Math.max(0, 14 - (now - scanStart - SCAN_MS) / 40);
    barMat.opacity = t < 1 ? 0.55 : Math.max(0, 0.55 - (now - scanStart - SCAN_MS) / 600);
    if (t >= 1 && !scanDone) {
      scanDone = true;
      setCounters(1);
    }
    if (t < 1) setCounters(eased);

    for (const cell of cells) {
      if (!cell.scanned && cell.x <= barX) setCellScanned(cell);
      const targetLift = cell.scanned && cell.leak ? 1 : 0;
      cell.lift += (targetLift - cell.lift) * 0.12;
      cell.tile.position.y = 0.32 * cell.lift;
      cell.tile.scale.y = 1 + 1.6 * cell.lift;
      if (cell.scanned && cell.leak) {
        cell.mat.emissiveIntensity = 0.6 + Math.sin(now / 320 + cell.x) * 0.18;
      }
    }

    // Zachte idle-rotatie + parallax
    const idle = Math.sin(now / 5200) * 0.05;
    group.rotation.y += (idle + targetRotY - group.rotation.y) * 0.06;
    group.rotation.x += (targetRotX - group.rotation.x) * 0.06;

    renderer.render(scene, camera);
  };
  requestAnimationFrame(animate);

  return {
    resize: setSize,
    setPointer: (x, y) => {
      targetRotY = x * 0.35;
      targetRotX = y * 0.18;
    },
    clearPointer: () => {
      targetRotY = 0;
      targetRotX = 0;
    },
    rescan: () => {
      cells.forEach(resetCell);
      bar.visible = true;
      barMat.opacity = 0.55;
      scanStart = performance.now();
      scanDone = false;
    },
  };
}
