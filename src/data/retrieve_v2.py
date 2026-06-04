#!/usr/bin/env python3
"""
NullState — Smart Data Retrieval v2

Three-pronged approach:
1. HNOCA: Resume Zenodo download (only ~2.5 GB left)
2. HDBCA + Cao Fetal: Use Census SOMA API to query cells directly from the
   cloud store — no need to download entire multi-GB files. We extract only
   the cell types we need for the reference.
3. Schema inspection for all successfully obtained data.
"""
import sys
import os
import yaml
import time
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

def load_config(config_path: str = 'config/paths.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ─── HNOCA: Resume Zenodo download ──────────────────────────────────────────
def resume_zenodo_download(url: str, dest: Path, chunk_size: int = 65536, 
                           max_retries: int = 500):
    """Resume download with larger chunks and aggressive retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            headers = {}
            mode = 'wb'
            initial_pos = 0
            
            if dest.exists():
                initial_pos = dest.stat().st_size
                if initial_pos > 0:
                    headers['Range'] = f'bytes={initial_pos}-'
                    mode = 'ab'
            
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    print(f"  ✓ Already fully downloaded: {dest}")
                    return True
                    
                r.raise_for_status()
                
                if r.status_code == 200 and initial_pos > 0:
                    # Server doesn't support Range — restart
                    mode = 'wb'
                    initial_pos = 0
                
                total = int(r.headers.get('content-length', 0)) + initial_pos
                
                with open(dest, mode) as f, tqdm(
                    desc=dest.name, total=total, initial=initial_pos,
                    unit='iB', unit_scale=True, unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)
            
            print(f"  ✓ Download complete: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return True
            
        except Exception as e:
            current_size = dest.stat().st_size / 1e9 if dest.exists() else 0
            print(f"  Retry {attempt+1}/{max_retries} ({current_size:.2f} GB so far): {e}")
            time.sleep(2)
    
    return False

# ─── HDBCA + Cao Fetal: Census SOMA query ────────────────────────────────────
def download_via_census_soma(collection_id: str, output_path: Path, 
                              organism: str = "homo_sapiens",
                              obs_filter: str = None,
                              max_cells: int = None) -> bool:
    """
    Use the Census SOMA store to extract an AnnData directly.
    This streams data from AWS S3 — much faster than downloading source h5ads.
    
    obs_filter: a string filter expression, e.g. 
        "tissue_general == 'brain' and is_primary_data == True"
    """
    try:
        import cellxgene_census
        import anndata as ad
        
        print(f"  Opening Census SOMA store...")
        with cellxgene_census.open_soma(census_version="2025-11-08") as census:
            # First, find dataset IDs for this collection
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()
            coll_ds = datasets[datasets["collection_id"] == collection_id]
            
            if coll_ds.empty:
                print(f"  No datasets found for collection {collection_id}")
                return False
            
            print(f"  Found {len(coll_ds)} datasets:")
            for _, row in coll_ds.iterrows():
                print(f"    - {row['dataset_title']} ({row.get('dataset_total_cell_count', '?')} cells)")
            
            # Get the dataset IDs
            dataset_ids = coll_ds["dataset_id"].tolist()
            
            # Build the value filter
            # Filter to only cells from this collection's datasets
            id_list = ", ".join(f"'{did}'" for did in dataset_ids)
            base_filter = f"dataset_id in [{id_list}]"
            
            if obs_filter:
                full_filter = f"{base_filter} and {obs_filter}"
            else:
                full_filter = base_filter
            
            print(f"  Query filter: {full_filter}")
            print(f"  Extracting AnnData from Census (this streams from S3)...")
            
            adata = cellxgene_census.get_anndata(
                census,
                organism=organism,
                obs_value_filter=full_filter,
            )
            
            if max_cells and adata.n_obs > max_cells:
                import numpy as np
                np.random.seed(42)
                idx = np.random.choice(adata.n_obs, max_cells, replace=False)
                adata = adata[idx].copy()
                print(f"  Subsampled to {max_cells} cells")
            
            print(f"  Got {adata.n_obs} cells × {adata.n_vars} genes")
            print(f"  .obs columns: {list(adata.obs.columns)}")
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Saving to {output_path}...")
            adata.write_h5ad(output_path, compression='gzip')
            print(f"  ✓ Saved ({output_path.stat().st_size / 1e9:.2f} GB)")
            return True
            
    except Exception as e:
        print(f"  Census SOMA query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    datasets_cfg = config.get("datasets", {})
    
    # === 1. HNOCA: Resume Zenodo download ===
    print("\n" + "="*60)
    print("  [1/3] HNOCA — Resuming Zenodo download")
    print("="*60)
    hnoca_cfg = datasets_cfg["hnoca"]
    hnoca_path = Path(hnoca_cfg["local_path"])
    
    if hnoca_path.exists():
        size_gb = hnoca_path.stat().st_size / 1e9
        if size_gb >= 17.4:  # Expected ~17.5 GB
            print(f"  Already complete ({size_gb:.1f} GB). Skipping.")
        else:
            print(f"  Partial download detected: {size_gb:.1f} GB / ~17.5 GB")
            print(f"  Resuming from Zenodo...")
            resume_zenodo_download(hnoca_cfg["zenodo_url"], hnoca_path)
    else:
        print(f"  Starting fresh download from Zenodo...")
        resume_zenodo_download(hnoca_cfg["zenodo_url"], hnoca_path)
    
    # === 2. HDBCA: Census SOMA query ===
    print("\n" + "="*60)
    print("  [2/3] HDBCA — Querying Census SOMA store")
    print("="*60)
    hdbca_cfg = datasets_cfg["hdbca"]
    hdbca_path = Path(hdbca_cfg["local_path"])
    
    if hdbca_path.exists() and hdbca_path.stat().st_size > 100_000_000:  # > 100 MB
        print(f"  Already downloaded ({hdbca_path.stat().st_size / 1e9:.1f} GB). Skipping.")
    else:
        # Remove the tiny 19 MB file from previous attempt
        if hdbca_path.exists():
            print(f"  Removing incomplete file ({hdbca_path.stat().st_size / 1e6:.0f} MB)")
            hdbca_path.unlink()
        
        success = download_via_census_soma(
            collection_id=hdbca_cfg["cellxgene_collection_id"],
            output_path=hdbca_path,
            obs_filter="is_primary_data == True",
        )
        if not success:
            print("  Retrying without is_primary_data filter...")
            download_via_census_soma(
                collection_id=hdbca_cfg["cellxgene_collection_id"],
                output_path=hdbca_path,
            )
    
    # === 3. Cao Fetal: Census SOMA query ===
    print("\n" + "="*60)
    print("  [3/3] Cao Fetal — Querying Census SOMA store")
    print("="*60)
    fetal_cfg = datasets_cfg["cao_fetal"]
    fetal_path = Path(fetal_cfg["local_path"])
    
    if fetal_path.exists() and fetal_path.stat().st_size > 100_000_000:
        print(f"  Already downloaded ({fetal_path.stat().st_size / 1e9:.1f} GB). Skipping.")
    else:
        if fetal_path.exists():
            fetal_path.unlink()
        
        success = download_via_census_soma(
            collection_id=fetal_cfg["cellxgene_collection_id"],
            output_path=fetal_path,
            obs_filter="is_primary_data == True",
        )
        if not success:
            print("  Retrying without filter...")
            download_via_census_soma(
                collection_id=fetal_cfg["cellxgene_collection_id"],
                output_path=fetal_path,
            )
    
    # === Schema inspection ===
    print("\n" + "="*60)
    print("  Running schema inspection")
    print("="*60)
    
    sys.path.insert(0, '.')
    from src.data.inspect_obs import inspect_all, save_schemas, print_schema_report, load_config as lc
    
    cfg = lc()
    schemas = inspect_all()
    if schemas:
        out_dir = Path(cfg.get("outputs", {}).get("schemas", "results/pilot/schemas/"))
        save_schemas(schemas, out_dir)
        print_schema_report(schemas)
    
    print("\n" + "="*60)
    print("  Phase 1 complete!")
    print("="*60)

if __name__ == '__main__':
    main()
