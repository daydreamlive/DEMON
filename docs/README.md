# DEMON Project Page

Source for **https://daydreamlive.github.io/DEMON/**. Served directly from this `/docs/` folder via GitHub Pages once the Pages toggle is flipped on (Settings → Pages → Source: `main` branch, folder: `/docs`).

## Preview locally

```bash
cd docs
python3 -m http.server 8080
```

Open `http://localhost:8080/`.

## Structure

```
docs/
├── .nojekyll              # tells GitHub Pages to skip Jekyll processing
├── README.md              # this file
├── index.html             # the whole page; one HTML file, no build step
└── assets/
    ├── css/main.css       # all styles (editorial-brutalist on pure black)
    ├── js/hero.js         # the hero canvas animation (six neon curves)
    ├── fonts/             # reserved for self-hosted webfonts (currently using Google Fonts CDN)
    ├── img/               # diagrams, og-image (currently empty, placeholders inline)
    └── video/             # demo + LoRA-morph videos (placeholders inline; replace with real .mp4/.webm)
```

## Replacing placeholders

The page currently has placeholder boxes in five spots. Replace each with a real `<video>` element:

- **`#demo` section**: the 30-second hero clip. Drop a file at `assets/video/hero-demo.mp4` and replace the placeholder block with `<video src="assets/video/hero-demo.mp4" autoplay muted loop playsinline></video>`.
- **`#showcase` section**: four LoRA-morph cards (deathstep / synthpop / ambient / techno). Same pattern in each `<div class="morph-card__media">`.

## Notes

- The page is intentionally a single static HTML file with one CSS file and one tiny JS file. No build step, no framework, no dependencies. Edits land instantly.
- Fonts currently load from Google Fonts CDN (IBM Plex Mono / Plex Serif / Plex Sans). Pre-launch we should switch to self-hosted woff2 files in `assets/fonts/` for archival reliability.
- The hero animation respects `prefers-reduced-motion`.
- Custom domain (e.g. `demon.daydream.live`) can be added later by dropping a `CNAME` file in this folder.
