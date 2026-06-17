# WS0a — Data Sourcing Findings

**Goal.** Resolve the three sourcing unknowns that gate Phase 1 and the Phase 2 bridge:
(1) where to get **true raw UMI counts for the HNOCA query** (the WS0a blocker that re-enables scArches surgery), (2) the **HEOCA** (endoderm) collection, (3) a **kidney/mesoderm** organoid dataset.
**Date:** 2026-06-12. Citations at the bottom. *I did not download anything — the actual fetch + the scArches NaN-test stay in your scvi/GPU environment.*

---

## 0. TL;DR

| Need | Status | Concrete action |
|------|--------|-----------------|
| HNOCA raw counts | **Strong lead** | Pull HNOCA from its **CELLxGENE** collection (the CELLxGENE schema *requires* raw counts) instead of the Zenodo `*_cleanedmeta.h5ad` the pilot used. Verify the layer is integer on download. |
| HEOCA (endoderm) | **Confirmed** | CELLxGENE collection `b4d13dc2-9b75-401d-9d9a-6d1468c17d90` — includes raw + normalized counts. ID now filled in `paths.yaml`. |
| Kidney (mesoderm) | **No single atlas** | Assemble from benchmarks; lead = Subramanian et al. 2019 census (450k cells, explicitly off-target-aware). Left as `TODO` in `paths.yaml` with leads. |

---

## 1. HNOCA raw counts — the WS0a blocker

**Why it matters (recap).** The pilot used `hnoca_cleanedmeta.h5ad` (Zenodo 14161275), which ships only log-norm + length-normalized layers. With no true integer UMIs the ZINB likelihood is mis-specified, so scArches surgery diverges to NaN and was disabled (`scarches_max_epochs: 0`). That leaves the query *projection-only*, inflating `ambiguous_novel` to 37.7% and undercounting `true_offtarget`.

**Primary lead — the CELLxGENE-hosted HNOCA object.** The CZ CELLxGENE Discover schema **requires raw counts** (in `X` or `raw.X`) for every submitted dataset. HNOCA is hosted on CELLxGENE (collection `de379e5f-52d0-498c-9801-0f850823c847`, already in `paths.yaml`), so the CELLxGENE copy should carry genuine integer counts that the cleaned Zenodo file dropped. **This is the lowest-effort fix: re-download the query from CELLxGENE rather than Zenodo, and confirm the count layer is integer.** Verify on download:

```python
import scanpy as sc, numpy as np, scipy.sparse as sp
a = sc.read_h5ad("hnoca_cellxgene.h5ad")
X = a.raw.X if a.raw is not None else a.X          # CELLxGENE puts raw counts in .raw or X
d = X.data if sp.issparse(X) else np.asarray(X).ravel()
print("integer counts?", bool(np.allclose(d[:200000], np.round(d[:200000]))), "max", float(d.max()))
```

If that holds, set `scarches_max_epochs: ~100` (the comment in `params.yaml` already flags this) and re-run the existing `_use_query_raw_counts` path in `train_scanvi.py` — it already auto-detects integer count layers.

**Most-correct fallback — per-study raw matrices.** HNOCA integrates **36 datasets / 1.77M cells**. Their original per-study raw UMI matrices (GEO/ArrayExpress/etc.) are the ground truth for surgery. The accession table is in the HNOCA paper's supplement and the reproducibility repo (`theislab/neural_organoid_atlas`). More work (per-dataset alignment to HNOCA barcodes), but yields unambiguous integer counts and is robust to any CELLxGENE re-normalization.

**Also check.** The *full* Zenodo record **14160929** ("Full Dataset") holds additional data representations/intermediate metadata beyond the cleaned file — worth a look for a retained counts layer.

**Recommended order:** CELLxGENE copy (verify integer) → if not clean, full Zenodo 14160929 → if still not, per-study GEO matrices.

---

## 2. HEOCA — endoderm (Phase 2) — CONFIRMED

**Atlas.** "An integrated transcriptomic cell atlas of human endoderm-derived organoids" (HEOCA) — Xu, Halle, Hediyeh-Zadeh, … Theis, Camp. *Nature Genetics* 57, 1201–1212 (2025). ~1M cells, 218 samples across lung, pancreas, intestine, liver, biliary, stomach, prostate. Same lab family as HNOCA.

**Data — has raw counts.** CELLxGENE collection **`b4d13dc2-9b75-401d-9d9a-6d1468c17d90`** explicitly provides *raw and normalized counts*, integrated embedding, cell-type annotations, and technical metadata. Also on Zenodo (8181495) and GitHub (`devsystemslab/HEOCA`); HCA portal: organoid-endoderm-v1-0.

**Action taken.** `paths.yaml` `heoca` slot updated with the confirmed collection ID + citation; `status: phase2_id_confirmed`. Because HEOCA ships raw counts, it will *not* hit the HNOCA surgery problem.

---

## 3. Kidney / mesoderm (Phase 2) — NO SINGLE ATLAS YET

Unlike neural (HNOCA) and endoderm (HEOCA), there is **no single consolidated CZI germ-layer organoid atlas** for kidney/mesoderm. The proposal's phrase "kidney organoid *benchmarks*" (plural) matches reality: the mesoderm slot must be **assembled** from benchmark datasets.

**Primary candidate (directly off-target-relevant).** Subramanian et al., "Single cell census of human kidney organoids shows reproducibility and diminished off-target cells after transplantation." *Nat Commun* 10, 5462 (2019). **450,118 cells**, 4 iPSC lines, compares organoid composition to human fetal/adult kidney and **explicitly quantifies off-target cells** — exactly the off-target mesenchyme question NullState asks. Data via GEO / Broad Single Cell Portal (SCP211). This is the strongest single starting dataset.

**Other candidates to fold in:** Wu et al. 2018 (Cell Stem Cell) comparative kidney-organoid scRNA benchmark; the 2023 PNAS kidney-organoid snRNA/snATAC multiome (gene-regulatory landscape of differentiation).

**Action taken.** `paths.yaml` `kidney_organoid` slot kept as `TODO` (no clean collection ID) with these leads recorded; treat kidney sourcing as an *integration* sub-task, not a one-file download.

---

## 4. Net effect on `paths.yaml`

- `heoca.cellxgene_collection_id` → **`b4d13dc2-9b75-401d-9d9a-6d1468c17d90`** (confirmed; raw counts present).
- `kidney_organoid.cellxgene_collection_id` → stays `TODO`; `paper`/notes point to the Subramanian 2019 census (GEO/SCP211) as the assembly seed.
- HNOCA query: no path change yet — the fix is to switch the *download source* to CELLxGENE and bump `scarches_max_epochs` once integer counts are confirmed (your GPU env).

---

## 5. Open verification steps (your environment)

1. Download HNOCA from CELLxGENE; run the integer-count check above. If clean → set `scarches_max_epochs: 100`, re-run mapping, measure how much `ambiguous_novel` migrates to `true_offtarget`.
2. Download HEOCA (`b4d13dc2-…`); confirm raw counts + the `origin`/germ-layer annotations needed for `expected_germ_layer: endoderm`.
3. Pull Subramanian 2019 kidney census from GEO/SCP211; decide whether one census suffices or several kidney benchmarks must be integrated for the mesoderm slot.

---

## Sources

According to PubMed (cite PubMed + DOIs):

- HEOCA — Xu Q, Halle L, Hediyeh-Zadeh S, et al. *An integrated transcriptomic cell atlas of human endoderm-derived organoids.* Nat Genet 57, 1201–1212 (2025). PMID 40355592. [DOI](https://doi.org/10.1038/s41588-025-02182-6)
- Kidney census — Subramanian A, Sidhom E-H, Emani M, et al. *Single cell census of human kidney organoids shows reproducibility and diminished off-target cells after transplantation.* Nat Commun 10, 5462 (2019). PMID 31784515. [DOI](https://doi.org/10.1038/s41467-019-13382-0)
- HNOCA — He Z, Dony L, Fleck JS, et al. *An integrated transcriptomic cell atlas of human neural organoids.* Nature 635, 690–698 (2024). [DOI](https://doi.org/10.1038/s41586-024-08172-8)

Web / data portals:

- HNOCA cleaned object (Zenodo 14161275): https://zenodo.org/records/14161275
- HNOCA full dataset (Zenodo 14160929): https://zenodo.org/records/14160929
- HNOCA reproducibility repo: https://github.com/theislab/neural_organoid_atlas
- HEOCA CELLxGENE collection: https://cellxgene.cziscience.com/collections/b4d13dc2-9b75-401d-9d9a-6d1468c17d90
- HEOCA HCA portal: https://data.humancellatlas.org/hca-bio-networks/organoid/atlases/organoid-endoderm-v1-0
- HEOCA repo: https://github.com/devsystemslab/HEOCA
- Kidney organoids atlas (Broad SCP211): https://singlecell.broadinstitute.org/single_cell/study/SCP211/human-kidney-organoids-atlas
