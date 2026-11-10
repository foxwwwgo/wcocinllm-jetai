# P1 Independent generation replication — frozen protocol

Purpose: evaluate whether the original concentration comparison is reproduced when candidates and self-confidence are independently regenerated. This is a follow-up replication planned after the original results and P0 diagnostics, not a previously preregistered study. No new benchmark response is collected before this protocol and code are hashed.

## Population and estimands

Use exactly the original 600 evaluation questions (300 GSM8K and 300 CommonsenseQA), without new development selection. Generate seven fresh candidates per question for deepseek-v4-flash, qwen3.8-max-0902 and glm-5.2. Commercial calls execute on the local machine. The original 0–100 scoring prompt, generation temperature 1, generation cap 2048, scoring temperature 0/cap 32, disabled thinking and evaluation parser remain unchanged. Do not request unsupported provider seeds or infer that equal model names imply equal weights. This is an independent collection batch, potentially including provider drift; it does not isolate seed randomness from endpoint drift.

Each candidate response is a new API request; no original response is copied into the new cache. Batch identity is `ida-p1-independent-20261024`. Score requests include the complete question and complete new visible response, without truncation. A valid original response may be reused only when resuming an interrupted request of THIS new batch. Distinct question/model/seat/kind/batch/payload identities prevent overwriting or cross-batch reuse. Raw request payloads and complete response bodies are committed before parsing. Credentials and authorization headers are excluded from records and publications.

H1: unconditional majority-overturning harm of W64 minus W4. H2: accuracy of G64 minus W64. Each uses all 1800 scheduled model–question records, first averages the three models within question, then weights the tasks equally. Questions, not candidates/model rows/batches, are the independent analysis units. Use the original stratified variance/Welch approximation and two-sided tests, Holm correction within these TWO replication hypotheses, and 10000 stratified whole-question bootstrap resamples (original numerical core and seed retained for comparability).

Report the new batch and original batch separately. H1 is independently supported only if its new-batch effect is positive and adjusted p<0.05; otherwise describe the effect and uncertainty without promoting a post-hoc endpoint. The original joint gate claim additionally requires H2 positive with adjusted p<0.05, correction retention at least 75%, positive H2 in at least four of six model–task cells covering all models and tasks, and at least 20 nonzero question-averaged H2 contrasts. Undefined retention fails. A result can support H1 without supporting the gate claim.

All fixed beta levels (0,1,2,4,8,16,32,64), G64, RANK4/64 and 200-per-pool paired permutation controls are retained. DEV uses the original development lock (all beta 0); no new temperature tuning is performed. Report C/H/J, all answer changes, valid/missing seats and scores, empty/single/multiclass pools and WQD by cell. Re-evaluate score diagnostics descriptively with the same definitions as P0.

Between-batch contrasts and the two-batch average are descriptive, with batches and models retained together within each question. They cannot replace an unsuccessful replication primary test. Two observed batches do not estimate the distribution of arbitrary future batches. New model-name aliases, request timestamps, returned model identifiers, response IDs, usage, finish reasons and elapsed time are preserved.

## Failure rules and operational stops

Keep original parser and fallback: invalid generation leaves a missing seat; do not replace formatting failures or truncated responses. Any valid candidate lacking confidence causes all score-dependent policies to return the original-pool majority. Empty pools abstain and count incorrect. Each logical request has up to three transport attempts for timeout/408/429/500/502/503/504 only, with the original exponential wait. No outcome-based resampling.

Bound concurrency at DeepSeek 4, Qwen 4, GLM 2, as in the original study. Stop new scheduling on 401/402/403, unsupported-request 400/404/422, five consecutive failed-transport bundles within a provider, or less than 512 MiB local free storage. Already issued requests may finish and are committed. Resume only the same frozen batch after an operational problem is resolved; record the interruption, do not overwrite records or silently change model IDs, parameters or parsers. Use a process lock to prevent duplicate collectors.

Complete all 12600 candidate seats once; at most 25200 logical generation/scoring requests, excluding permitted transport retries. Monitor progress, HTTP errors, finish reasons, format rates and disk availability, but do not inspect partial correctness, choose subsets or stop on effects. Analyze only after completion. No fees are estimated.

## Sample-size interpretation and reproducibility

Six hundred independent questions preserve the original scope; this is not a guarantee of power for a 0.556-point effect. Under a simple paired-binary approximation with two-sided alpha .025, 80% power and discordance 0.01/0.02/0.05, a 0.556-point effect would require roughly 3,100/6,200/15,400 independent units; model averaging and covariance can change these values. Hence direction, event counts and intervals must accompany significance. These planning calculations do not change sample size or stopping.

Release the frozen protocol and hashes, complete parsed records, full response export with hashes, deterministic offline calculations, figures/tables and interruption history. The raw collection folder remains local. Any substantive amendment is separately timestamped before subsequent collection and cannot retrospectively make the old run match the new definition.
