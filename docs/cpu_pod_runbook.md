# CPU-pod runbook — 16 vCPU / 64 GB RAM / 100 GB disk

For the CPU+GPU split: a cheap CPU pod does the I/O- and CPU-bound work; the H100 pod does only
the training hours. Both mount the **same RunPod network volume** so data passes between them.

## Verdict: does 16 vCPU / 64 GB / 100 GB work?

**Yes for the CPU pod's proper jobs — with one caveat (the reference build).**

| Task | Fits 64 GB / 100 GB? | Notes |
|------|----------------------|-------|
| `smoke_test_cpu.py` | ✅ trivially | synthetic ~3k cells |
| `fetch_verify_cellxgene.py` (HNOCA) | ✅ | ~17 GB download (disk fine); verify is backed (low RAM) |
| `run_aim3.py` (WS2 geometry) | ✅ | runs on `z_lin` (a small `.npy`); CPU, low RAM |
| `geosketch_subsample.py` **`--backed --use-rep`** | ✅ | your `--backed` mode loads only selected cells |
| `geosketch_subsample.py` from-scratch **PCA** | ⚠️ | PCA over the full atlas needs ≫64 GB → use `--use-rep` (see ordering) |
| **Reference build** (`run_pilot.py --skip-to-step 1`) | ⚠️ **the one risk** | concat of ~2 M cells peaks ~40–60 GB; full Cao pull is ~4 M cells |

So every **script I handed you is ready** for this pod. The only piece that doesn't comfortably
fit 64 GB is the harmonized-reference build — handled below.

## The reference-build caveat (and two clean fixes)

The build concatenates HDBCA (~1.6 M) + a Cao-fetal subset; `download_census.py` /
`retrieve_v2.py` currently materialize the **full primary** Cao atlas (~4 M cells) in memory
before subsetting — that alone can exceed 64 GB and a big chunk of 100 GB. Pick one:

- **(A) Recommended — build the reference on the GPU pod.** H100 pods ship with far more RAM;
  make the build + HVG the first ~1 h there. Keeps the CPU pod trivially within 64 GB/100 GB and
  sidesteps the Cao-schema details. Costs ~1 h of GPU time.
- **(B) Keep it on the CPU pod — cap Cao first.** Cao only provides mesodermal/endodermal/
  neural-crest *anchors*, so a capped sample is scientifically fine (same logic as
  `dish_max_cells_per_type`). Pull Cao with a cell cap (e.g. `download_via_census_soma(..., max_cells=150000)`),
  **delete the raw inputs after building**, then the standard in-memory build fits ~30–40 GB.
  First run `python scripts/inspect_obs.py` to confirm Cao's lineage column before relying on the
  `Main_cluster_name` filter in `run_pilot.py` step 1 (Census obs may differ).

> Want option (B) turnkey? I can add a `--low-ram` on-disk-concat build + a Cao cap to the
> retrieval — say the word and I'll wire it (it needs one real-data test pass to confirm the
> Cao obs schema).

## Disk budget (100 GB)

If the CPU pod only does download + verify + geometry + geosketch:

| File | ~Size |
|------|------|
| HNOCA (CELLxGENE, raw counts) | ~17 GB |
| z_lin / latents / sketch outputs | ~1–3 GB |
| **CPU-pod total** | **~20 GB** ✅ comfortable |

If you also build the reference on the CPU pod (option B): add HDBCA ~12 GB + Cao **capped** ~2–8 GB
+ reference output ~11 GB → ~50–60 GB total. Fits 100 GB **only if Cao is capped and raws are
deleted after building**; the full Cao pull will not.

## CPU-pod runbook (in order)

```bash
# 0. env (CPU wheels)
pip install "scvi-tools>=1.0" torch --index-url https://download.pytorch.org/whl/cpu
pip install anndata scanpy scikit-learn geosketch requests cellxgene-census

# 1. validate the code path (free, ~2 min) -- do this BEFORE any GPU spend
PYTHONPATH=. python scripts/smoke_test_cpu.py

# 2. WS0a: fetch HNOCA from CELLxGENE + confirm integer raw counts
PYTHONPATH=. python scripts/fetch_verify_cellxgene.py --out-dir data/
#    -> if it reports integer counts, set params.yaml scarches_max_epochs: 100

# 3. reference: prefer option (A) on the GPU pod; or option (B) here with a capped Cao.

# --- after the GPU pod has trained scVI and written X_scVI into an .h5ad on the shared volume ---

# 4. geosketch on CPU (low-RAM): needs a precomputed embedding, so run AFTER scVI
PYTHONPATH=. python scripts/geosketch_subsample.py \
    --input data/reference_with_scvi.h5ad --output data/reference_sketch150k.h5ad \
    --n-cells 150000 --use-rep X_scVI --backed

# 5. WS2 geometry on CPU once z_lin (contrastiveVI background latent) exists
PYTHONPATH=. python scripts/run_aim3.py --zlin z_lin.npy --labels populations.csv \
    --out results/pilot/aim3_geometry.json
```

## Ordering gotcha (geosketch + geometry need GPU outputs first)

`geosketch --use-rep X_scVI` and `run_aim3 --zlin …` both consume artifacts the **GPU** produces
(the scVI latent; the contrastiveVI `z_lin`). So the CPU pod's independent pre-GPU work is really
steps 1–2 (smoke-test + fetch/verify); steps 4–5 run on the CPU pod **after** the GPU pod writes
its latents to the shared volume. That's still the cost win: downloads + geometry stay off the H100.

## Bottom line

16 vCPU / 64 GB / 100 GB is the right size for the CPU pod's role — **as long as the
reference build runs on the GPU pod (option A) or uses a capped Cao (option B).** The four CPU
scripts are ready as-is for this hardware.
