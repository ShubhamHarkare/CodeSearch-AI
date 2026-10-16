"""Cleans one react.dev MDX page into plain markdown. Pure function, no I/O.

What goes here:
    - ``CleanPage`` (dataclass): title (from frontmatter), body (markdown).
    - ``clean_mdx(raw: str) -> CleanPage`` which:
        * parses and removes the YAML frontmatter, keeping ``title``
        * drops heading-anchor comments like ``{/*use-state*/}`` (keep the
          anchor id: urls.py needs it)
        * unwraps ``<Sandpack>`` blocks so the inner code fences remain
        * turns ``<Pitfall>``, ``<Note>``, ``<DeepDive>`` into labelled text,
          e.g. "Pitfall: ..."
        * removes purely visual components (``<Intro>``, ``<Diagram>``,
          ``<InlineToc/>``) but keeps their text
        * never edits text inside code fences
    - small private helpers, one per transformation.

Must not:
    - read files or know about chunks or URLs.

Tested by: tests/unit/test_mdx_clean.py with real snippets from the mini
corpus (Sandpack unwrap, Pitfall label, code fence untouched, frontmatter title).
Phase: 1
"""
