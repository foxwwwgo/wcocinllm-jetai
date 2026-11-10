# Manuscript figure and table style

The five manuscript figures share the settings in `analysis/figure_style.py`. The additional `followup_gate` graphic is a release-only illustration and is not part of the five-figure manuscript.

## Geometry and typography

- Every final vector PDF has a 160 mm canvas width, matching the manuscript text width without reduction. Figure heights vary with the content.
- DejaVu Sans is used throughout. Panel headings are 9 pt bold; standard labels are 8.5 pt; tick and model labels are 7.5 pt. The dense counts in Figure 5 use 7 pt. Legends use 8 pt in a shared row above the panels.
- PDFs retain vector content and embedded fonts; PNG previews are 300 dpi. The export checks text bounds before saving and records the result in `figure_style_manifest.json`.

## Color and non-color cues

Corrections use green (#009E73); harms use orange-red (#D55E00) with hatching; net changes use charcoal (#263238), including diamond markers in Figure 5. Blue (#0072B2) marks comparisons or structural emphasis. Missing opportunities use gray. The model curves in Figure 3 use a fixed four-color palette with distinct markers and line styles. Colors are supplemented by labels, position, hatching and symbols, so readers can interpret grayscale copies.

## Tables

All four manuscript tables retain black text and horizontal booktabs rules, without vertical borders or colored cells. Captions are above the tables. General explanatory notes appear below in smaller type and begin with italic `Note:`. The prefix is a consistent editorial convention; the IDA guide does not prescribe that literal label.

## Scientific content

The graphical revision preserves the data, recorded uncertainty intervals, policy definitions and statistical conclusions. The five CSV exports in `figure_data/` remain byte-identical to v10.4. Figure 5 independently checks its 48 policy cells against `tables/all_readouts.csv`. No model requests or new experiments are part of this formatting revision.
