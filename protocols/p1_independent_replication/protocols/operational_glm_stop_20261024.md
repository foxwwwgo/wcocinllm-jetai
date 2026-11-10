# GLM billing stop and cache-aware resumption

At 2026-10-24 13:54:32 UTC the frozen collector stopped because GLM had five consecutive failed transport bundles. Inspection of transport errors identified HTTP 429 / code 1113 (insufficient balance or no available resource package). Six logical GLM requests exhausted their permitted three attempts: four generation requests and two confidence requests. All eighteen denial bodies are retained in their original cache directories. In-flight work was committed before exit.

The completed-seat cache contains 2,617 DeepSeek, 1,051 Qwen and 594 GLM records. The GLM count includes six transport-failed seats. The final status counter shows 593 for GLM because the last in-flight record was committed after the stop was triggered. The cache count, not that counter, determines what remains to be scheduled. No partial correctness or hypothesis result has been inspected.

All six exhausted transport failures remain in the scientific records and are NOT retried or replaced. Missing generations and missing confidence use the originally frozen missing-seat and majority-fallback rules. This preserves the three-attempt limit. Successful generated answers and scores also remain unchanged.

The original runner replays cached failures into its consecutive-failure counter on every restart, causing the same historical stop again. The separately hashed `src/resume_after_transport_stop.py` wrapper therefore schedules only seats without a completed parsed record. It counts existing records toward completion but applies the consecutive-failure stop to newly attempted bundles. It calls the unchanged frozen per-seat collector, preserves the same schedule order, concurrency, retry limits, models, prompts, parser, fallback, and technical stopping thresholds. All thirteen originally frozen files remain unchanged. This is an operational cache-resumption correction, recorded before any resumed collection; it does not change a scientific endpoint or resample an outcome.

Resume only after the GLM account issue is resolved. Retain every interruption and transport attempt in the final release. No fees are estimated.
