# License scope

This release uses different licenses for different materials; it is not a dual-license choice for every file.

| Material | Applicable grant |
|---|---|
| Original Python code in `analysis/` and `protocols/p1_independent_replication/src/`, and original software configuration | MIT, in `LICENSE`; pre-existing third-party material is excluded |
| Original explanatory documents, original experimental annotations, derived numerical results, tables, figure data, and standalone plot assets in `figures/` | CC BY 4.0, in `LICENSE-DATA` and `LICENSES/CC-BY-4.0.txt`, to the extent of the authors' rights |
| Benchmark questions, options, reference answers and benchmark passages embedded in any record | The applicable upstream benchmark terms; see `THIRD_PARTY_NOTICES.md` |
| CISC prompt text and any other reproduced CISC expression | Apache 2.0; retain the Google Research Authors attribution and included Apache license |
| Complete request/response logs and generated model text | Mixed-content records: original annotations, organization and derived results receive the CC BY grant; third-party text and provider/model content are not blanket relicensed |
| `documentation/manuscript/**`, `sections/*.tex`, and any other manuscript LaTeX/BibTeX or article/supplement PDF | Separately handled under `MANUSCRIPT_RIGHTS.md`; no MIT or CC BY grant is applied to the manuscript by this release |
| License texts and upstream notices | Preserve their own terms and attribution; they are not authored or relicensed by this project |

The manuscript exception takes precedence over the general documentation rule. Standalone plot PDFs in `figures/` are figure assets, not full article or supplement PDFs. The code grant covers original code even where it implements a cited mathematical definition; it does not override notices for copied third-party expression.

In mixed files, the grant is component-specific, not a claim to own all strings in a JSON object. Question text, choices, reference-answer text, quoted benchmark solutions, CISC prompt passages, provider response bodies and model-generated prose retain their respective upstream rights and applicable terms. Original prompts not borrowed from upstream sources, annotations, derived labels and numerical analyses are covered only to the extent the authors can grant those rights. Facts and other non-protected material do not acquire new restrictions through this notice.

Frozen experimental files are deliberately unchanged. For example, the historical statement about awaiting author license review in `protocols/cisc/study_protocol.json` records the status at the experimental freeze, not a current pending license selection. `LICENSE_SELECTION.md` records the present decision. No frozen hash or experiment record has been rewritten to add license headers.

`documentation/license_scope.json` maps every supplied file to a scope category. For mixed-content files the category covers only the original contribution; the exclusions above continue to apply. 
