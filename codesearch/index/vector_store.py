"""Small wrapper around Chroma. The only module that imports chromadb.

What goes here:
    - ``VectorStore(path, collection_name)`` using ``chromadb.PersistentClient``
      with cosine distance.
    - ``upsert(chunks, vectors)``, ``delete(ids)``, ``all_ids() -> set[str]``,
      ``count() -> int``, ``get(ids) -> list[Chunk]``.
    - ``query(vector, k) -> list[tuple[Chunk, float]]`` returning cosine
      SIMILARITY (convert from Chroma's distance) in rank order.

What it must do:
    - Store Chunk fields as metadata (url, anchor, title, header_path as a
      joined string) and rebuild Chunk objects on the way out.

Tested by: tests/unit/test_vector_store.py in tmp_path (upsert is idempotent,
delete works, query order, count).
Phase: 1
"""
