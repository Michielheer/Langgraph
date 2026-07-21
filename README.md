# House of Intelligence — website

Marketing-/bedrijfssite voor House of Intelligence B.V. (Breda): insurtech- en
AI/data-consultancy voor het Nederlandse volmacht- en verzekeringsdomein, met
**Leaklight** (premielek-detectie tijdens polisconversie) als vlaggenschip.

## Stack

- [Astro](https://astro.build) 5 — statische site
- [Tailwind CSS](https://tailwindcss.com) 4 (via `@tailwindcss/vite`)
- [Three.js](https://threejs.org) voor de 3D live-scan in de hero — draait in een
  Web Worker op een OffscreenCanvas zodat de main thread vrij blijft (Lighthouse
  performance 100); main-thread fallback voor oudere browsers, statisch 2D-grid
  zonder WebGL
- Vanilla JS voor scroll-reveal en kaart-tilt — geen verdere libraries
- Fonts: Inter (display + body-fallback), Helvetica Neue voor bodytekst

## Ontwikkelen

```bash
npm install
npm run dev       # http://localhost:4321
npm run build     # productie-build naar dist/
npm run preview   # bekijk de productie-build lokaal
```

## Deploy

Statische output in `dist/` — direct te hosten op Vercel, Netlify of elke
statische host. Geen server of environment-variabelen nodig.

## Design-tokens

Vastgelegd in `src/styles/global.css` (`@theme`): licht cloud-wit canvas
(`--color-cloud`), witte kaarten, één periwinkle accent (`--color-periwinkle`)
voor alle interactieve momenten. In de live-scan is periwinkle de scanbalk,
**coral** het "lek gevonden"-signaal en **mint** "gedekt". Pill-buttons (9999px),
16px kaartranden en violet-getinte schaduwen volgen het RevenueCat-register.
