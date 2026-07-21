# House of Intelligence — website

Marketing-/bedrijfssite voor House of Intelligence B.V. (Breda): insurtech- en
AI/data-consultancy voor het Nederlandse volmacht- en verzekeringsdomein, met
**Leaklight** (premielek-detectie tijdens polisconversie) als vlaggenschip.

## Stack

- [Astro](https://astro.build) 5 — statische site
- [Tailwind CSS](https://tailwindcss.com) 4 (via `@tailwindcss/vite`)
- Vanilla JS voor de live-scan animatie en scroll-reveal — geen verdere libraries
- Fonts: Bricolage Grotesque · Inter · JetBrains Mono (Google Fonts)

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

Vastgelegd in `src/styles/global.css` (`@theme`): petrol-achtergrond (`--color-ink`),
amber als "lek gevonden"-signaal, mint als "gedekt". Cijfers en labels staan
altijd in JetBrains Mono.
