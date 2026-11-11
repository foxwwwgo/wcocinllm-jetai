# Expanded-population study: collection notebook

This directory records the collection as it was run. The authoritative description of the
repository and the table/figure mapping is the top-level `README.md`.

---

# Cross-family scoring and conformal filtering experiment 

## Purpose

Test whether the conformal certificate from §3.2 is usable when the **signal is stronger**:
expand the decision unit from 24 `W64` rewrites to 209 competition pools, and compare the
discriminative power and the correct-fix vs. harm trade-off of four scoring signals.

## Environment

- Collection host: single V100-32GB, 214 GB RAM
- Inference: llama.cpp `llama-server` (bundled in e9_runtime), OpenAI-compatible API
- Models: `Phi-3-mini-4k-instruct-q4`, `qwen2.5-14b-instruct-q5_k_m` (both weights already present on the server)
- Items: `openai/gsm8k` test split, `tau/commonsense_qa` validation, downloaded via hf-mirror

## Data

- Decision unit: 209 competition pools in `certificate_pool_ledger.json` (600 questions x 3 hosted models)
- Candidate answers: taken directly from the ledger's `counts` (parsed results of 7 paths), not regenerated
- Question-id mapping: `gsm8k-test-<i>` -> i-th row of test split; `csqa-<32hex>` -> `id` of CommonsenseQA validation
- Validation: 2,048 (record, candidate class) tasks, 0 missing item text, 0 gold mismatches

## Four signals

| Signal | Construction |
|---|---|
| Self-confidence | Per-path self-score in ledger, averaged within cluster |
| Phi-3-mini judge | Question + candidate answer -> 0-100 probability |
| Qwen2.5-14B judge | Same as above |
| Qwen2.5-14B solver | Model solves independently; signal is "its answer == candidate answer" |

Decision rule: reassign to the highest-scoring non-majority cluster iff its score exceeds the majority cluster's.

## Results

| Signal | AUC | Reassign | Corrected C | Harmed H | eta | 95% CI | sign test p |
|---|---:|---:|---:|---:|---:|---|---:|
| Self-confidence | 0.572 | 69 | 20 | 40 | 0.50 | [0.25, 0.90] | 0.013 |
| Phi-3-mini judge | 0.598 | 36 | 10 | 16 | 0.62 | [0.21, 1.50] | 0.33 |
| Qwen2.5-14B judge | 0.688 | 56 | 23 | 27 | 0.85 | [0.41, 1.75] | 0.67 |
| Qwen2.5-14B solver | 0.671 | 59 | 25 | 29 | 0.86 | [0.42, 1.75] | 0.68 |
| R1-Distill-7B solver | 0.573 | 54 | 16 | 33 | 0.48 | [0.21, 0.95] | 0.021 |
| judge ∧ solver | — | 28 | 15 | 11 | 1.36 | [0.50, 4.60] | 0.56 |
| qwen solver ∧ r1 solver | — | 21 | 11 | 9 | 1.22 | [0.38, 4.67] | 0.82 |
| two-of-three signals agree | — | 31 | 19 | 11 | 1.73 | [0.71, 6.00] | 0.20 |

Upper bound: 55 pools can be corrected, 135 pools carry harm risk.

**Conformal coverage** (5-fold cross-fit by question, counting only records whose candidate pool
contains the correct class): self-confidence 0.958 / 0.902 / 0.815 (alpha = 0.05 / 0.10 / 0.20);
Qwen judge 0.950 / 0.946 / 0.946. All reach the nominal level.

**The certificate's marginal contribution to selectivity is zero**: intersecting "score comparison"
with "challenger passes certificate" leaves Phi and Qwen's eta unchanged (0.62, 0.85), because scores
saturate near 100 and almost all challengers get certified.

## Conclusions

- Self-confidence rewriting is **significantly harmful** (eta = 0.50, p = 0.013).
- Switching to a cross-family scorer roughly doubles eta (0.50 -> 0.85), monotonically tied to
  discriminative power, but still below 1.
- When two independent signals agree simultaneously eta = 1.36, the only configuration above 1, but
  the interval [0.50, 4.60], p = 0.56, is not yet established.
- The conformal certificate itself holds (coverage target met), but the selectivity bottleneck is the
  signal, not the bound.

## Files

- `out/scores_phi.jsonl`, `out/scores_qwen14b.jsonl`: candidate answer scoring
- `out/solve_qwen14b.jsonl`: independent solving results
- `out/summary.json`: summary
- `scripts/`: task-building, scoring, analysis, and summary scripts

## R1-Distill failure and fix

On first run, all 2,048 scoring entries failed to parse. The cause was not an unusable model: as a
reasoning model it first emits `reasoning_content`, and under `max_tokens=16` the `content` is always
empty; once the cap was lifted it solved the problem directly instead of judging.

Fix: stop treating it as a judge and instead use it as an **independent solver** -- `max_tokens=1024`,
with fallback to the backup field (`reasoning_content` when `content` is empty), parsing
`Proposed answer:` plus a multi-level fallback regex. All 600 questions ran through, 25 empty results
(4.2%), GSM8K standalone accuracy 80.7%.

Conclusion: used alone, the R1-Distill signal is no better than self-confidence (AUC 0.573, eta 0.48,
p = 0.021, significantly harmful); but conjoined with the Qwen solver eta = 1.22, same direction as
judge∧solver's 1.36, and the two share no common training lineage.

## Scale-up experiment

- Volume: GSM8K 1,200 + CommonsenseQA 1,200 = 2,400 questions (4x the original 600)
- Generators: granite-3.3-2b, Phi-3-mini-4k, each 7 paths + per-path confidence
- Scale: 16,800 x 2 sequences
- Written to disk: `out/gen_granite_1200.jsonl`, `out/gen_phi_1200.jsonl`

## Fifth signal: DeepSeek API 

Two request configs, `deepseek-chat` and `deepseek-reasoner`, each used for judging and independent
solving. The vendor resolves both to the same model, DeepSeek-V4.1-Flash (identifier `deepseek-flash`),
corresponding to the non-thinking and thinking modes respectively. The table labels below have been
unified to the model name as "DeepSeek ... (non-thinking) / (thinking)"; the item text is identical to
the local signals (`tasks.jsonl`, 2,048 candidate-scoring tasks / 600 questions).

| Signal | candidate AUC | % scored 100 | Reassign | Corrected C | Harmed H | eta | eta_adj | 95% CI | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Self-confidence | — | — | 69 | 20 | 40 | 0.50 | 0.25 | [0.25, 0.90] | 0.013 |
| Phi-3-mini judge | 0.621 | 28.0% | 36 | 10 | 16 | 0.62 | 0.31 | [0.21, 1.50] | 0.33 |
| Qwen2.5-14B judge | 0.669 | 0.2% | 56 | 23 | 27 | 0.85 | 0.43 | [0.41, 1.75] | 0.67 |
| Qwen2.5-14B solver | — | — | 59 | 25 | 29 | 0.86 | 0.43 | [0.42, 1.75] | 0.68 |
| DeepSeek judge (non-thinking) | 0.543 | 85.6% | 8 | 1 | 2 | 0.50 | 0.25 | [0.00, 3.00] | 1.00 |
| DeepSeek solver (non-thinking) | — | — | 48 | 18 | 23 | 0.78 | 0.39 | [0.34, 1.62] | 0.53 |
| DeepSeek judge (thinking) | **0.855** | 76.3% | 47 | 22 | 18 | **1.22** | 0.61 | [0.58, 2.45] | 0.64 |
| DeepSeek solver (thinking) | — | — | 41 | 19 | 14 | 1.36 | 0.68 | [0.58, 4.14] | 0.49 |
| Qwen∧DeepSeek solver (non-thinking) | — | — | 21 | 13 | 6 | 2.17 | **1.08** | [0.71, 10.0] | 0.17 |
| Qwen∧DeepSeek solver (thinking) | — | — | 17 | 10 | 5 | 2.00 | **1.00** | [0.60, 12.0] | 0.30 |

Three conclusions:

1. **Discriminative power and decision utility are decoupled**. The thinking-mode judge's candidate-level
   AUC 0.855 is the highest of all ten signals, yet 76.3% of candidates are scored 100, allowing only 47
   reassignments with eta just 1.22. Good ranking is not the same as being usable as a switch.
2. **Saturation is model behavior, not item presentation**. CSQA candidates given only the letter are
   scored 100 96.0% of the time; adding the option text still yields 95.3%. The non-thinking mode scores
   AUC 0.712 on arithmetic items and 0.553 on multiple-choice; the item-type difference is the main cause.
3. **The only configuration stably above eta_adj = 1.0 remains the dual-solver conjunction** (1.08 / 1.00),
   but the interval [0.71, 10.0], p = 0.17, with a sample of 21 reassignments, is only directional evidence.

### Solver capability and token budget

A 400-token cutoff pushes the non-thinking solver's GSM8K accuracy down to 86.0%; relaxing to 1500
restores it to 88.7%, and the standalone probe subset (n=300) reaches 93.3%. The thinking-mode solver at
4000 tokens reaches GSM8K 97.0%, CSQA 88.8%, the strongest solver in the whole set. Judging is likewise
affected by truncation: at 512 tokens the thinking mode produced 214 entries (10.4%) whose output was
exhausted by reasoning with no conclusion; relaxing to 4096 and re-running dropped the empty count to 74.

### Reproduction files

- `scripts/dscore.py`: API judging/solving, controlled by `DS_MODEL`, `DS_JUDGE_MT`, `DS_SOLVE_MT`
- `scripts/dbreak.py`: signal comparison split by GSM8K / CSQA
- `scripts/dfinal.py`: summary with cluster-bootstrap CI and sign test
- `scripts/dprobe.py`: three robustness probes (token budget, prompt wording, option presentation)
- `out/scores_deepseek.jsonl`, `out/solve_deepseek1500.jsonl`, `out/scores_dsreasoner.jsonl`, `out/solve_dsreasoner.jsonl`
- `out/deepseek_summary.json`

## Parameter-count ladder 

Purpose: add a third isomorphic point, Qwen2.5-14B-Instruct, to the two in §5.8 (2B granite, 3.8B Phi),
so that "parameter count cannot endogenously rescue self-confidence" shows a trend rather than a single
straight line segment.

- Design: same 2,400 questions (1,200 GSM8K + 1,200 CSQA), 7 paths per question, temperature 1, 512 token
  cap, per-path self-confidence, fully isomorphic to granite/Phi
- Output: `out/gen_qwen14b_1200.jsonl` (16,800 entries)
- Throughput: measured 0.53 entries/sec, total 8 hours 10 minutes (14B q5 aggregates ~150 tokens/sec on V100)
- Wrap-up: after generation completes, run `scripts/ladder.py`, which writes `out/ladder.txt`
- **Result** (`out/ladder.txt`, all 16,800 entries parseable):

| Generator | Params | Competition pool | Majority-correct | Reassign | C | H | eta | eta_adj | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite-3.3-2B | 2B | 1,391 | 54.5% | 36 | 0 | 23 | 0.00 | 0.00 | 0.513 |
| Phi-3-mini-4k | 3.8B | 1,153 | 62.8% | 418 | 81 | 263 | 0.31 | 0.15 | 0.585 |
| Qwen2.5-14B-Instruct | 14B | 439 | 63.6% | 103 | 31 | 55 | 0.56 | 0.28 | 0.647 |

- Interpretation: eta rises monotonically with scale (0.00 -> 0.31 -> 0.56), but the increment shrinks
  (+0.31, +0.25); 14B still has only eta_adj 0.28, 3.5x short of 1.0. The number of competition pools
  falls 3x with scale, showing part of the improvement comes from "fewer corrigible opportunities".
- This table has been merged into the main text §5.3 (Table 3)

**Warning**: `build_expand.py` builds pools with `glob("out/gen_*_1200.jsonl")`. If `build_expand.py`
is re-run after the 14B generation completes, `pools_expanded.json` becomes a 7,200-pool three-model
version, and all numbers in existing §5.8 change accordingly. The ladder analysis deliberately only
reads the gen files and does not modify `pools_expanded.json`, so the two do not interfere. To upgrade
the main table to three models, you must synchronously update the summary, §5.8, Table 6, and Figure 5.
