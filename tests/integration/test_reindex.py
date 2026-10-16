"""Live re-indexing tests.

Cases: edit one mini-corpus page -> build_index reports only that page's chunks
as changed; the new sentence becomes retrievable; a snapshot taken before the
swap still sees the old version; two concurrent triggers never overlap.
"""
