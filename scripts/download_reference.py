#!/usr/bin/env python
"""Stage the harmonized-reference source atlases from the CELLxGENE Census, RAM-safely.

Census ``get_anndata`` returns RAW integer counts in X and the CL ontology ``cell_type`` (exactly
what build_reference.py + origin_map.py expect). HDBCA is the neural core (all primary). Cao fetal
supplies the NON-neural off-target anchors and is filtered to the origin_map's mesoderm/endoderm/
neural_crest cell_type labels AT THE CENSUS LEVEL, so we never materialize the full ~4M-cell atlas.

    PYTHONPATH=. python scripts/download_reference.py [--census-version 2025-11-08]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.retrieve_v2 import download_via_census_soma, load_config
from src.utils.origin_map import get_default_origin_map


def main():
    ap = argparse.ArgumentParser(description="Stage HDBCA + Cao reference atlases from the Census")
    ap.add_argument("--only", choices=["hdbca", "cao", "both"], default="both")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    ds = cfg["datasets"]

    # Non-neural CL ontology labels (from the origin map) to pull from Cao.
    om = get_default_origin_map()
    nonneural = sorted({k for k, v in om.items() if v in ("mesoderm", "endoderm", "neural_crest")})
    ct_list = ", ".join(f"'{c}'" for c in nonneural)
    cao_filter = f"is_primary_data == True and cell_type in [{ct_list}]"
    logging.info(f"Cao non-neural anchor cell_types ({len(nonneural)}): {nonneural}")

    import anndata as ad

    def _staged_ok(path: Path) -> bool:
        """A download counts only if the file exists and has >0 cells (backed, cheap)."""
        if not path.exists():
            return False
        try:
            a = ad.read_h5ad(path, backed="r")
            n = a.n_obs
            try:
                a.file.close()
            except Exception:
                pass
            return n > 0
        except Exception as e:
            logging.error(f"staged file {path} unreadable: {e}")
            return False

    failures = []
    if args.only in ("hdbca", "both"):
        logging.info("=== HDBCA (neural core) ===")
        ok = download_via_census_soma(
            ds["hdbca"]["cellxgene_collection_id"], Path(ds["hdbca"]["local_path"]),
            obs_filter="is_primary_data == True",
        )
        if not (ok and _staged_ok(Path(ds["hdbca"]["local_path"]))):
            failures.append("hdbca")

    if args.only in ("cao", "both"):
        logging.info("=== Cao fetal (non-neural anchors) ===")
        ok = download_via_census_soma(
            ds["cao_fetal"]["cellxgene_collection_id"], Path(ds["cao_fetal"]["local_path"]),
            obs_filter=cao_filter,
        )
        if not (ok and _staged_ok(Path(ds["cao_fetal"]["local_path"]))):
            failures.append("cao_fetal")

    if failures:
        logging.error(f"Reference staging FAILED for: {failures}")
        sys.exit(1)
    logging.info("Reference atlas staging complete.")


if __name__ == "__main__":
    main()
