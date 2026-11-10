# Third-party notices

Original project code is MIT-licensed and original data/documentation is CC BY 4.0-licensed only within `LICENSE_SCOPE.md`. The following upstream rights remain in force wherever the corresponding content appears, including inside requests, responses, JSON strings, protocol descriptions and explanatory documents.

## GSM8K

- Source: https://github.com/openai/grade-school-math
- License: MIT. Copyright (c) 2021 OpenAI.
- The complete upstream notice is preserved verbatim in `LICENSES/GSM8K-MIT.txt`.
- Scope here: retained GSM8K question/answer/solution passages and reproductions within prompts or responses. The project does not claim these benchmark materials as its own original data.

## CommonsenseQA

- Source: https://www.tau-nlp.org/commonsenseqa
- Dataset card: https://huggingface.co/datasets/tau/commonsense_qa
- Attribution: Alon Talmor, Jonathan Herzig, Nicholas Lourie and Jonathan Berant, *CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge*, NAACL-HLT 2019. https://doi.org/10.18653/v1/N19-1421
- The dataset card declares MIT. `LICENSES/CommonsenseQA-NOTICE.md` records that declaration, its source and the standard MIT permission/warranty text without inventing an upstream copyright holder or year.
- Scope here: retained question text, choices, reference-answer material and reproductions inside prompts/responses. This is the original CommonsenseQA benchmark, not ECQA or a separately licensed explanation dataset.

## CISC and its released confidence prompts

- Official source: https://github.com/taubenfeld/CISC
- Used commit: `40011d9745662612ec125683eae47a1e6572b0bb`.
- Copyright 2025 The Google Research Authors. Apache License, Version 2.0.
- Full license: `LICENSES/Apache-2.0.txt`. The original collection's attribution notice is preserved as `LICENSES/CISC-ORIGINAL-NOTICE.md`; its reference to Python files concerns the original collection directory, not a claim that those files are bundled here.
- The release retains CISC prompt passages in `protocols/cisc/study_protocol.json` and `data/cisc_complete_responses.jsonl.gz`. Borrowed prompts and any reproduced upstream expression retain Apache 2.0. The release does not replace that license with MIT or CC BY.
- The original matched experiment used upstream prompt text, answer-prefix extraction and aggregation semantics with a locally adapted hosted-model runner. Local sample selection, failure handling, statistics and the present offline decision checks are documented separately. No Google internal runner or full upstream CISC source tree is bundled.
- Method attribution: Taubenfeld et al., *Confidence Improves Self-Consistency in LLMs*, Findings of ACL 2025, https://doi.org/10.18653/v1/2025.findings-acl.1030 . The question-balanced, tie-adjusted WQD variant is documented in `documentation/WQD_definition.md`.

## Model services, generated text and dependencies

The request/response archives retain outputs from the services and local models identified in the frozen protocols and `data/study_provenance.json`. They are experimental records, not model weights or an authorization to use a service account. Credentials, authorization headers, model weights and inference binaries are not supplied. Access to a service and third-party rights in its outputs remain governed by applicable provider/model terms; this package does not purport to waive those terms or warrant ownership of generated prose. The CC BY grant applies to the authors' original annotations, derived results and compilation rights, where such rights exist.

Python, NumPy, SciPy, Matplotlib and other dependencies referenced by configuration files are installed separately and retain their own licenses. A dependency name or methodological citation does not place the upstream software under this project's license.

Upstream author and organization names in these notices identify the third-party sources and are intentionally preserved in both release variants. They are not the identities of this manuscript's authors. See `documentation/license_sources.json` for license-source receipts checked on 2026-10-25.
