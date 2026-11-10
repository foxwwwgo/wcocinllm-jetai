# Figure 5: six-model recovery–harm overview

The graphic uses three row blocks and two task columns. The initial block displays DeepSeek-V4.1-Flash, qwen3.8-max-0902, glm-5.2 and Qwen2.5-14B-Instruct. The two added-family blocks display Granite-3.3-2B-Instruct and Phi-3-mini-4k-instruct under evaluation and compatibility parsing, respectively.

Each configuration has W4, W64 and G64 rows. Green bars are corrected majority errors; orange hatched bars are newly introduced errors plotted to the left; charcoal diamonds and the Net column report C−H. The axis scale is shared across all six panels. Zero-change rows are retained.

All counts are recomputed from stored predictions and references, checked against recorded correctness and the complete readout CSV. Each cell retains 300 scheduled questions. The source hash manifest names the three unchanged row files. This figure reorganizes existing results and adds no experiment, tuned policy, significance test or new independent sample.

The manuscript's initial primary inference, data partition, parser definitions and stopping decisions remain unchanged. The figure adds a common visual comparison across configurations, with the parser distinction explicit.

The figure uses the same 160 mm canvas width, font family, legend size and semantic palette as the other manuscript figures; see `figure_style.md`.

The companion `figure_data/figure_5_pool_composition.csv` reports usable competition for every configuration. Granite and Phi have 49 and 11 multiclass evaluation-parsed pools, versus 128 and 144 compatibility-parsed pools. Task rows locate both evaluation-parser H1 changes in Granite CommonsenseQA and the ten compatibility-parser changes across all four model–task cells. These descriptive counts supplement the common-denominator outcome bars; they do not change accuracy denominators or create independent questions.

The three blue block headings are vertically separated from the task panel titles.
