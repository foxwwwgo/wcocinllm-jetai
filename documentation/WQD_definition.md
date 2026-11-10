# WQD: source and reported variant

Source: Amir Taubenfeld, Tom Sheffer, Eran Ofek, et al. *Confidence Improves Self-Consistency in LLMs*. Findings of ACL 2025, pp. 20090–20111. DOI: https://doi.org/10.18653/v1/2025.findings-acl.1030. The definition is Section 6, Definition 6.1 (PDF page 6). Official source: https://aclanthology.org/2025.findings-acl.1030/.

The original definition pools every correct–incorrect candidate pair within questions and reports the proportion for which the correct candidate has a strictly greater score. A question with more pairs contributes more weight; ties contribute zero.

The present article uses **question-balanced, tie-adjusted WQD**. For each eligible question, average 1(correct_score > incorrect_score) + 0.5*1(correct_score == incorrect_score) over its correct–incorrect pairs; then average question values equally. Eligibility requires at least one correct and one incorrect valid candidate and complete scores for all valid candidates. Empty, unmixed or score-incomplete pools do not enter this diagnostic. Missing generation seats alone do not disqualify a pool. No eligible question gives an undefined estimate; a single eligible question has no interval.

Main Sections 2.2 and 5.3 and Supplementary Section S2.3 state this variant. Figure 4, its intervals and all reported WQD estimates use this definition. The weighting and tie treatment differ from the original; results are not directly interchangeable with original CISC WQD. WQD is adapted from prior work, not a new metric contribution.

Implementation: analysis/confidence_diagnostics.py, function score_cell. It appends one half-tie comparison mean per eligible question and averages this list. The published independent-batch diagnostic is data/independent_replication/confidence_quality.json. A legacy per-row wqd field in frozen output can compare only available scores in an incomplete pool; it does not replace the reported complete-pool diagnostic. Frozen files and result values are unchanged.

This update clarifies provenance and definitions; no experimental scores, policies or WQD values are changed.
