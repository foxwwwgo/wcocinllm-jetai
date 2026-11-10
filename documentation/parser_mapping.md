# Parser names and preserved record keys

The current manuscript uses three distinct names. Their implementations in `analysis/frozen_parsers.py` are unchanged. Frozen protocols, row fields and diagnostic category keys retain their historical spelling.

| Manuscript name | Function | Rule for generation | Accepted added-model seats |
|---|---|---|---:|
| Strict | `parse_strict` | Exactly one matching parenthesized `Proposed answer: (X)` anywhere in the response; no last-line requirement. | 501 |
| Evaluation | `parse_v71` | Apply strict parsing first. On failure, check the last nonempty line for a proposed-answer marker, stripping asterisks/backticks and allowing optional parentheses. | 877 |
| Compatibility | `parse` | Apply evaluation parsing first. On failure, allow last-line `Answer:` or `Final answer:` forms, or a terminal `(X=value)` form. | 2,842 |

All three require HTTP 200 and a normal `stop`/`eos` finish. Generation values must match the task-specific letter/integer grammar. The confidence branch uses the shared finite 0–100 number rule. A failed strict parse, including multiple strict matches, can still succeed through the evaluation fallback. The released code and twenty boundary cases specify the exact grammar and precedence.

`python analysis/verify_parser_mapping.py` replays all 8,400 complete added-model outputs without reference answers and writes `data/parser_mapping.json`. The accepted-seat counts by task/model are:

| Task | Model | Strict | Evaluation | Compatibility |
|---|---|---:|---:|---:|
| CommonsenseQA | Granite-3.3-2B-Instruct | 421 | 441 | 673 |
| CommonsenseQA | Phi-3-mini-4k-instruct | 53 | 54 | 516 |
| GSM8K | Granite-3.3-2B-Instruct | 8 | 192 | 621 |
| GSM8K | Phi-3-mini-4k-instruct | 19 | 190 | 1,032 |

## Dataset and figure mapping

- `data/evaluation_rows.json`: `answers` uses evaluation parsing; `strict_answers` supplies the initial strict sensitivity.
- `data/replication_rows.json`: `answers` and `original_answers` use evaluation parsing; `strict_answers` uses strict parsing; `extended_answers` uses compatibility parsing.
- `data/replication_extended_parser_rows.json`: `answers` and `extended_answers` use compatibility parsing; the alternative `original_answers` and `strict_answers` retain the evaluation and strict outputs.
- `data/development_rows.json`: the separate-question study uses compatibility parsing for `answers` across all six configurations. Alternative arrays retain their named meanings.
- `data/parser_outputs.jsonl.gz`: `original_answer` means evaluation and `extended_answer` means compatibility.
- `data/parser_audit.json` and its failure ledger: category `accepted_original` means accepted by the evaluation parser. The boundary-case field `original` likewise means evaluation.
- Figure 5 blocks `initial` and `added_original` use evaluation parsing. `added_compatibility` uses compatibility parsing. The readout-table key `replication_original` and pool-summary key `replication_original_parser` also mean evaluation. Table 4 lists evaluation / compatibility in that order.

The matched CISC study has its own released answer extraction, described separately in Appendix C. These three names refer to the common concentration-study parsing rules; they do not replace CISC's implementation.
