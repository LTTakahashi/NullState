# TMLR resubmission notes

## What happened
Submission 10714, *"No Overlap, No Invariance: An Identifiability Boundary for
Anchor-Free Disentanglement, and the Functionals That Survive It"*, was **desk
rejected** (not sent for review). Format was not the problem: 13pp, unmodified
`tmlr.sty`, correctly anonymized. The rejection was the AE-judgment kind —
"insufficient evidence… quality of writing… or limited interest".

Diagnosis (probable, in order):
1. **A general claim on n=1 evidence.** An *identifiability boundary* is a general
   statement; it was tested on one dataset pair, with the key quantity (support
   overlap) observed at one value rather than varied.
2. **The headline statistic was confounded.** The >=0.998 adversary cannot separate
   a *removable* batch effect from genuine support mismatch — our own later work
   proved this (it fires at ~1.0 on the harmless large-delta case).
3. **Shape was a chain of negatives**: fails -> fallback fails -> weaker thing
   partly survives.
4. **Biology vocabulary throughout**, reading as an application case study →
   "limited interest within the TMLR audience".

Old paper preserved at `main_v1_rejected.tex.bak` and in git at `a56f6f2`.

## What replaces it
`main.tex` is now a **different paper**, not a revision. Two of the old paper's
claims have been superseded by our own later work, so revising it would mean
defending results we have since retracted.

| | rejected | current |
|---|---|---|
| evidence | 1 dataset pair | ~1,000 controlled runs, gated DGP, 25 seeds |
| core claim | asserted boundary | closed-form derivation + measured invariance classes |
| audience | single-cell integration | anyone using MCC / DCI / kNN-R^2 |
| rigor | — | pre-registration, power analysis, TOST, closed-form null |

## Before resubmitting — checklist
- [ ] Check https://openreview.net/forum?id=qL8FE0AU1o for an AE comment; it beats
      inference about why.
- [ ] Declare the prior submission in the OpenReview **"Previous TMLR Submission
      URL"** field. Concealing it is a hard reject.
- [ ] Confirm `\usepackage{tmlr}` has NO `[accepted]`/`[preprint]` option.
- [ ] Re-run the anonymization scan (see below) on the final PDF.
- [ ] Post the preprint regardless — costs nothing, establishes date. Zenodo DOI
      already serves this; arXiv needs endorsement (try verifying an institutional
      email on the arXiv account first — that often auto-endorses cs.LG/stat.ML).
- [ ] **Create the anonymized artifact and replace `ANON-PLACEHOLDER` in main.tex.**
      Mirror the repo at https://anonymous.4open.science (paste the GitHub URL, set
      an expiry past the review period) and paste the generated link into the
      Reproducibility Statement. Do NOT link the real repo or the Zenodo record.
- [ ] If the *same* paper is on Zenodo under your real name, that is a permitted
      preprint, but do not cite it in the submission and do not add a
      "code available at ..." line pointing to it.

```bash
python3 -c "
from pypdf import PdfReader
t=' '.join(p.extract_text() for p in PdfReader('main.pdf').pages)
print([w for w in ['Takahashi','Washington','WSU','Voiland','NullState'] if w.lower() in t.lower()] or 'clean')"
```

## Reproduce the figures
```bash
python make_figs_v3.py     # reads probe-demo/results_*.csv
tectonic -X compile main.tex
```
