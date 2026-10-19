# Mini corpus for tests

22 pages copied unchanged from the React docs repository
([reactjs/react.dev](https://github.com/reactjs/react.dev)) at commit
`046f17d04295ba047bb5739026b4ac3110f17028` (the same commit pinned in
`codesearch/config.py`). Paths are kept relative to `src/content/`, so
`reference/react/useState.md` here is `src/content/reference/react/useState.md` upstream.

The React documentation is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0); the full license text is in `LICENSE-DOCS.md` in this folder.
Copyright © Meta Platforms, Inc. and affiliates.

## Why these pages

| What it exercises | Pages |
| --- | --- |
| `<Sandpack>` blocks to unwrap | `reference/react/useState.md` (14), `reference/react/useEffect.md` (17), `learn/responding-to-events.md` (15), and most `learn/` pages |
| `<Pitfall>` / `<Note>` / `<DeepDive>` labels | `reference/react/Component.md` (6 pitfalls, 22 notes), `reference/react/useMemo.md`, `reference/react/memo.md` |
| Very long pages (chunk splitting) | `reference/react/Component.md` (~1,950 lines), `reference/react/useEffect.md` (~1,860 lines) |
| Short pages | `reference/react/lazy.md`, `reference/react/createContext.md` |
| `index.md` -> folder URL | `reference/react/index.md`, `learn/index.md` |
| `react-dom` and nested paths | `reference/react-dom/createPortal.md`, `reference/react-dom/components/input.md` |
| Topics from the v1 test questions | useState, useReducer, useEffect, useMemo, useCallback, context, memo, lazy, portals, error boundaries (`Component.md`), forms (`input.md`), Server Components |

Do not edit these files: tests compare against the real upstream text.
