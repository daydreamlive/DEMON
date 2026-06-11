# DEMON docs

Public-facing documentation site for DEMON, built with
[Mintlify](https://mintlify.com). Same tooling as the Daydream docs at
`pipelines/apps/docs`.

## Local development

```bash
cd apps/docs
npm install
npm run dev
# open http://localhost:3033
```

Navigation, theming, and redirects live in `docs.json`. Pages are MDX with
YAML frontmatter. Screenshots live in `images/`.

The one-off development notes in the repo-root `docs/` folder are *not* part
of this site — this app is the public documentation; `docs/` is internal.

## Checks

```bash
npm run broken-links
```
