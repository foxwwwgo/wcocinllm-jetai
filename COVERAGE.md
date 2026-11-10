# Release coverage

| Study | Included records | Offline reproduction |
|---|---:|---|
| Initial concentration evaluation | 2,400 model–question rows / 600 questions | Deterministic readouts; 200-per-pool permutation controls; paired counts; cell and whole-question-cluster intervals and tests; complete-case and strict-parser sensitivities |
| Independent generation | 1,800 model–question rows / same 600 questions | Full new-batch readouts and H1/H2; retained original development lock; descriptive same-question batch contrasts; full requests/responses and interruption history |
| Added families, evaluation parser | 1,200 rows / same 600 questions | All readouts, development locks, permutation controls, pooled statistics and pool coverage |
| Added families, compatibility parser | 1,200 rows / same 600 questions | The same analysis with the separately locked parser |
| Added-model generated outputs | 8,400 complete texts | Strict, evaluation and compatibility parsing, 20 boundary cases, syntax census and 80-output fixed sample |
| Separate-question development | 1,200 rows / 200 questions | All twelve rules, exact decimal boundary, ranking, selection and stopping decision |
| Matched CISC evaluation | 600 rows / 600 questions | Decisions, paired bootstrap intervals, exact McNemar tests, secondary Holm adjustment and opportunity counts |

Repeated model rows and parser variants do not create additional independent questions. Models remain together within each question during the task-stratified bootstrap. The added families reuse the initial question set. The new-question screen has no collected pilot or reserved validation outcomes.

## Source and reproduction boundaries

The initial study supplies 39,182 complete request/response records (19,600 generation and 19,582 confidence requests). CISC supplies 14,700 (4,900 generation, 4,900 P(True), 4,900 verbal). `initial_complete_responses.jsonl.gz` and `cisc_complete_responses.jsonl.gz` include complete retained response bodies, credential-free request payloads, finish/usage fields and response IDs. The exporter and offline verifier check body and canonical request-identity hashes. Authentication headers are excluded. Original endpoint aliases cannot authenticate undisclosed fixed model weights.

`parser_outputs.jsonl.gz` retains all 8,400 complete added-model evaluation texts, parsed targets and content/body hashes. The original HTTP envelopes are not included for that study. Full response exports do not include the separate-question development screen, which remains reproducible from parsed records and its frozen specification. No pilot or validation observations are claimed for that stopped screen.

Statistical replay reproduces stored policy decisions and question-cluster estimates. Hosted-model regeneration depends on sampling and provider state. The completed P1 collection is separately frozen and analyzed only after all scheduled seats are committed. Its H1/H2 tests are separate from the original batch. Descriptive between-batch comparisons retain both batches and all models within each of the same 600 question clusters.

## Added descriptive analyses

Sixteen model-task-parser score-quality cells; question-bootstrap WQD/Brier intervals; fixed-bin reliability data; all C/H/J counts and pool composition; initial development grids; 10,000 CISC development bootstrap selections and evaluation decision stability without evaluation labels. `data/cisc/development_rows.json` supplies the complete 100-question input for temperature reselection. A single eligible WQD question has no interval.

## Effective competition

The 1,800 primary records contain 209 multiclass pools: 27 have constant scores and 182 have nonconstant scores. At beta=4 the certificate covers 26 and 106, respectively; at beta=64 it covers 26 and zero. These counts describe the mathematical condition's application, not a new sampled population.

Each added configuration has 600 scheduled evaluation questions. With evaluation parsing, Granite contributes 49 multiclass pools and Phi 11 (60 total); with compatibility parsing they contribute 128 and 144 (272 total). The companion composition export includes empty and single-class counts and task-level outcome contributions. All 600 questions per configuration remain in accuracy denominators, and models share question IDs.

## Independent-generation transport coverage

`independent_generation_complete_responses.jsonl.gz` contains 25,193 distinct logical requests and 25,237 transport attempt bodies, including archived billing denials. Seven exhausted GLM transport failures remain in their originally parsed records: six during the billing interruption and one subsequent congestion failure. Successful candidates and scores were not replaced. The cache-aware resume wrapper and its separate hash manifest preserve all thirteen originally frozen files. See `data/independent_replication/transport_accounting.json` and the operational records for counts and timing. These resource measurements do not estimate charges.

## Manuscript organization and WQD

The main article retains all five figures and four tables; full proofs and protocols are in separate Supplementary Material. WQD uses the question-balanced, tie-adjusted CISC-derived variant specified in documentation/WQD_definition.md. All numerical files, analysis code, frozen protocols and plotted values remain unchanged.
