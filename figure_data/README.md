# Figure source data

Manuscript: *When Confidence Overrides Consensus in Large Language Models*, v10.15.

The CSV files map directly to the five manuscript figures. Figure 1 is a conceptual workflow and has no experimental data series. The numerical series are regenerated from the recorded pools and freshly recomputed statistics.

| Figure | File | Contents and units |
|---|---|---|
| 2(a,c) | `figure_2_recovery_harm_batches.csv` | Original and independent-batch corrections, harms and net change for W4, W64 and G64; 1,800 model–question records per batch. |
| 2(b,d) | `figure_2_effects_batches.csv` | Separate batch effects and marginal 95% question-cluster intervals in percentage points; 600 independent questions per batch, the same question IDs across batches. |
| 3 | `figure_3_concentration.csv` | Every fixed concentration for all eight initial model/task cells; accuracy change in pp and harm as a percentage of all 300 questions. Positive beta values use log2 spacing, with zero shown separately. |
| 4 | `figure_4_opportunity.csv` | Counts partitioning majority errors into correct-class-present and correct-class-absent pools; each cell has 300 questions. The label `o` is the correct-class-present count. |
| 4(c,d) | `figure_4_score_quality.csv` | Within-question discrimination, descriptive question-bootstrap intervals, eligible-question counts, and Brier loss. |
| 4(e,f) | `figure_4_reliability.csv` | Ten fixed probability bins; each score-available question contributes equal total weight. Marker area increases with bin mass. |
| 5 | `figure_5_six_model_recovery_harm.csv` | All 48 model/parser/task/policy cells, with corrections, harms, net count and net pp. Harms are displayed to the left as negative values but stored as nonnegative counts; net = corrections − harms. |
| 5 companion | `figure_5_pool_composition.csv` | Empty, single-class and multiclass pool counts by model, task and parser, with H1/H2 count contributions. The 24 rows include combined-task totals; do not sum those totals with their task rows. This companion data does not add a figure or a formal manuscript table. |

Figure 5 keeps the historical block keys `initial`, `added_original` and `added_compatibility`. The first two use the evaluation parser; the third uses the compatibility parser. `added_original` does not denote the strict parser. Model rows share task question IDs. Parser variants use different usable pools from the same scheduled questions; these rows are not independent replications. All scheduled questions remain in denominators, including failures.

Recreate with `python analysis/run_all.py` from the release root, which runs statistics before figure export. The figure generators are `analysis/recompute_diagnostics.py` (Figures 1 and 3 plus the original-only tradeoff), `analysis/independent_replication_figure.py` (the current four-panel Figure 2), `analysis/score_quality_figure.py` (the complete Figure 4), and `analysis/create_six_model_overview.py` (Figure 5). `manifest.json` records source hashes and row counts. No API credentials, requests or GPU are needed.

The older `figure_2a_recovery_harm.csv` and `figure_2b_effects.csv` are retained as original-only provenance. They are not the complete numerical inputs to the current Figure 2. No partial new-batch observations are used.

Figure 2 displays Delta_H (harm) and Delta_gap (accuracy). These are H1 and H2 in the unchanged frozen numerical files and CSV endpoint keys; both retain their original predefined-test role. G64-M remains descriptive. This label mapping changes presentation only.
