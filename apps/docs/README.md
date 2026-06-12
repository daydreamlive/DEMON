# DEMON docs

Public-facing documentation site for DEMON, built with
[Mintlify](https://mintlify.com). Same tooling as the Daydream docs at
`pipelines/apps/docs`.

This directory is the product documentation source: install guides, demo
operator docs, API references, troubleshooting, and deployment notes. The
repo-root `docs/` folder is a separate GitHub Pages project/paper landing page.

## Local development

```bash
cd apps/docs
npm install
npm run dev
# open http://localhost:3033
```

Navigation, theming, and redirects live in `docs.json`. Pages are MDX with
YAML frontmatter. Screenshots live in `images/`.

## Checks

```bash
npm run build
npm run broken-links
```
