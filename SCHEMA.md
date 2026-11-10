# Core record schema

`evaluation_rows.json`, `replication_rows.json`, `replication_extended_parser_rows.json` and `development_rows.json` are arrays with one row per model and question. Null answer denotes an invalid/missing seat; null prediction denotes abstention. All scheduled questions remain represented.

- `id`, `task`, `model`: stable question ID, task and internal model key.
- `answers`, `scores`: aligned seat arrays; scores use 0–100 before normalization. Invalid answer seats are excluded equally for all readouts. Missing scores on valid seats cause majority fallback.
- `gold`, `majority`, `top`, `gap`: reference class, first-occurrence plurality, tied leading classes and top-two count difference.
- `predictions`: deterministic readout decisions; `correct`: corresponding 0/1 outcomes. PERM entries in `correct` are average correctness over fixed score permutations.
- `candidate_n`, `complete`, `score_constant`, `wqd`, `ess`: pool usability and descriptive score diagnostics.
- `strict_answers` denotes strict-parser outputs. In the initial rows, `answers` uses evaluation parsing. In added-model raw records, `original_answer` uses evaluation parsing and `extended_answer` uses compatibility parsing. Historical names are retained; see `documentation/parser_mapping.md` for file-specific mappings.

Model keys: `deepseek` = DeepSeek-V4.1-Flash (called through the provider's `deepseek-flash` identifier); `qwen` = qwen3.8-max-0902; `glm` = glm-5.2; `open_qwen14b` = Qwen2.5-14B-Instruct; `granite` = Granite-3.3-2B-Instruct; `phi` = Phi-3-mini-4k-instruct. Configurations, returned identifiers and weight/runtime provenance are recorded separately.

`data/cisc/evaluation_rows.json` uses `prediction` (singular), `ptrue` for binary-normalized probability, `verbal` for 0–100 confidence, `confidence_complete` per branch, and `O`, `E`, `E_minus`, `E_plus` for opportunity indicators. CISC normalization temperatures are stored in `protocols/cisc/temperature_lock.json`.

`data/successor_development_lock.json` contains every ranked candidate, thresholds and the decision. Candidate keys `b16-h2-d0.10`, for example, identify concentration 16, maximum count gap 2 and minimum class-mean confidence gap 0.10. Exact rational comparison applies to that last boundary.

## Figure 5 export

`tables/six_model_recovery_harm.csv` has 48 rows: eight model/parser configurations × two tasks × three policies. The six model names include Granite and Phi twice, once per parser. Fields are `block`, internal `model`, displayed `model_name`, `task`, `policy`, `scheduled_questions`, `corrections`, `harms`, `net`, and `net_pp`. `net = corrections - harms`; `net_pp = 100 * net / 300`. The displayed negative harm bars represent `-harms`, while the CSV keeps harm counts nonnegative.

## Statistical outputs and mechanism diagnostics

`recomputed_statistics.json` contains freshly computed initial/extension cells, question-cluster contrasts, strict-parser sensitivity, CISC intervals/tests, sign-symmetry enumeration and descriptive model exclusions. `monte_carlo`, `environment`, `source_sha256` and comparison tolerances specify replay. Exclusion analyses have no new multiplicity-adjusted confirmatory status.

`mechanism_diagnostics.json` separates single-class and multiclass pools, constant scores, certificate coverage and realized overrides. `certificate_pool_ledger.json` provides per-pool classification. Examples are chosen by lowest salted SHA-256 within correction, harm and certified-multiclass-preservation strata; their stored counts, scores and stabilized class totals explain each decision.

The historical Figure 5 block key `added_original` denotes evaluation parsing, as does `initial`; `added_compatibility` denotes compatibility parsing. No block in Figure 5 uses the strict parser.

## Full-output parsing

`parser_outputs.jsonl.gz` is gzip-compressed UTF-8 JSON Lines, with one full generated response per seat. Fields include `question_id`, `task`, `model`, `seat`, `generation_key`, `original_answer`, `extended_answer`, `http_status`, `finish_reason`, `raw_reply`, `raw_reply_sha256` and source artifact/body hashes. It excludes request prompts, headers and credentials. Model-produced text may contain benchmark wording.

`parser_audit.json` contains a gold-free syntax census and boundary-test count. The precedence is evaluation acceptance (historical category `accepted_original`), compatibility-only acceptance, then rejection classes: transport/finish failure, empty text, multiple explicit proposed-answer markers, recognized answer marker outside the grammar, or no accepted marker. The last two occur in these data. A recognized marker is not necessarily a valid final answer; absence of the accepted marker is not evidence of no substantive answer. Counts do not estimate reasoning accuracy.

`parser_failure_ledger.json` maps every generation key to its category. `parser_review_sample.json` retains 20 rejected texts per task/model cell selected by a fixed hash. `parser_suffix_observations.json` records qualitative assistant inspection of the sample endings, explicitly without human-gold or accuracy claims. `parser_boundary_tests.json` supplies 20 executable cases, including final-line syntax, multiple markers, invalid options, units, trailing text, confidence parsing and abnormal finish/status behavior.

## Score variation, scale controls and effective competition

`extended_diagnostics.json` supplies the current nonconstant-score preservation example, a 14-row certificate grid (seven positive beta values × two score strata), four scale-control summaries, two observed decision identities, 24 pool-composition rows, and retention interpretation flags. The analysis is descriptive. Source hashes identify the parsed input rows and the unchanged statistical output.

`tables/certificate_score_strata.csv` uses all primary multiclass records, stratified by whether scores on valid candidates are constant. `certified` requires complete scores and a unique plurality satisfying beta × normalized-score-span < log(top-count / second-count). `overrides` counts actual deviations from original-pool majority. Failure of certification does not imply an override.

`tables/rank_scale_readouts.csv` reports primary corrections and harms for W4, W64, RANK4 and RANK64; ranks are zero-based midranks divided by n−1. `observed_identities` counts equal predictions on the initial 2,400 records only, not equivalence outside those records or between untested parameter values.

`tables/pool_composition.csv`, also exported as `figure_data/figure_5_pool_composition.csv`, contains eight model/parser configurations, each with two task rows and one combined row. Empty, single-class and multiclass counts sum to all scheduled records. Combined rows duplicate their task components and must not be added to them. `H1_net_harms` is the W64−W4 harm-count difference; `H2_net_correct` is the G64−W64 correct-count difference. Model keys follow the mapping above. These strata do not replace the primary population or accuracy denominators.

`tables/retention_interpretation.csv` labels boundary-degenerate empirical bootstrap intervals. An interval of [1,1] can arise when all observed corrections are retained; it is not a population guarantee. `undefined_resamples` counts bootstrap draws with no associated correction opportunity and hence undefined retention. Existing estimates are preserved, including those undefined counts.
