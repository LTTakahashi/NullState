# TMLR submission build

Identifiability-framed version of the NullState work, targeted at Transactions on Machine
Learning Research (TMLR).

## Build
```bash
tectonic main.tex     # -> main.pdf (anonymized submission version)
```

## Anonymization (critical)
`\usepackage{tmlr}` with **no options** produces the anonymized submission version
("Under review as submission to TMLR"). TMLR is double-blind and **non-anonymous submissions
are rejected without review**. Do not add `[accepted]` or `[preprint]` until acceptance.

The submitted PDF must contain no author name, affiliation, email, or links to named
artifacts (the GitHub repo and Zenodo record both carry the author name). The build is
checked for these leaks before submission.

## Relationship to the biology manuscript
`docs/manuscript/` holds the organoid-QC framing (bioRxiv/Zenodo). This directory holds the
identifiability framing for TMLR. Same underlying results and figures, reordered:

| TMLR section | Content | Figure |
|---|---|---|
| Setup | overlap boundary condition | - |
| Benchmark | the near-disjoint 4.4M-cell testbed | fig1, fig2 |
| Result 1 | anchor-free invariance not identified (adversary 0.998) | fig4 |
| Result 2 | translation-invariant functionals remain estimable | fig5 |
| Result 3 | precondition tested: ordering survives, distances do not | fig5 |
| Result 4 | reference incompleteness vs genuine divergence | fig3, fig6 |

TMLR permits preprints (arXiv/bioRxiv); dual submission only bars archival peer-reviewed venues.

## Pending before submission
- The lineage-decodability probe (what else \zlin retains besides domain).
- Optional but high-value: a controlled overlap sweep locating the identifiability transition.
