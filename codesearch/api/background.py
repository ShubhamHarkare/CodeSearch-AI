"""Work that runs beside requests inside the same process (no queues, no workers).

What goes here:
    - ``ReindexRunner(settings, embedder, handle, cache)``:
        * ``asyncio.Lock`` for single-flight: a second trigger while one runs
          is coalesced into "run once more after this one"
        * ``trigger(reason)``: called by the webhook, the timer and the watcher
        * runs ingest/build.py in a thread (``asyncio.to_thread``), then logs the
          IngestReport and updates Prometheus counters
    - ``periodic(interval_hours)``: asyncio task calling trigger("schedule").
    - ``watch_docs(path)``: watchdog observer on a local docs folder, debounced
      (e.g. 5 s), calling trigger("file-change").
    - ``start(app)`` / ``stop(app)`` used by the lifespan in main.py.

Tested by: tests/integration/test_reindex.py (two triggers -> builds never
overlap; file change -> new chunk retrievable).
Phase: 3
"""
