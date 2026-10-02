# flipbook web

React + TypeScript GUI over the `flipbook` API (`/api/*`).

```bash
npm install
npm run dev      # vite dev server, proxies /api to :8484
npm run build    # tsc + vite build -> dist/ (served by `flipbook serve`)
npm run lint     # oxlint
```

Run the backend against the demo store:

```bash
flipbook serve --store fixtures/demo_store
```
