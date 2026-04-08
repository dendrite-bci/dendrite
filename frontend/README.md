# Dendrite v2 — Frontend

Vue 3 SPA for real-time EEG/BCI control, visualization, and ML training.

## Stack

- Vue 3 with `<script setup lang="ts">`, Pinia stores, composables
- PrimeIcons, Tailwind CSS, uPlot for real-time charts
- Vite dev server with API proxy to FastAPI backend

## Development

```bash
npm install
npm run dev          # Vite dev server on :5173, proxies /api and /ws to :8321
```

Backend must be running on port 8321 (see root CLAUDE.md for commands).

## Production Build

```bash
npx vite build       # Outputs to dist/
```

The built `dist/` directory is served by FastAPI automatically — no separate web server needed. See root CLAUDE.md for deployment instructions.

## LAN Access (dev mode)

```bash
npm run dev -- --host    # Exposes on 0.0.0.0:5173
```
