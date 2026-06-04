#!/usr/bin/env python3
"""
Fast alternative data retrieval for NullState.
Downloads from CELLxGENE Census/Discover API (AWS-backed, much faster than Zenodo).
Preserves the existing partial Zenodo download untouched.
"""
import sys
import os
import json
import yaml
import shutil
import requests
from pathlib import Path

def load_config(config_path: str = 'config/paths.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ─── Strategy 1: CELLxGENE Discover API (presigned S3 URLs) ─────────────────
def get_collection_datasets_via_api(collection_id: str) -> list[dict]:
    """Query the CELLxGENE Discover API for datasets in a collection."""
    url = f"https://api.cellxgene.cziscience.com/dp/v1/collections/{collection_id}"
    print(f"  Querying Discover API: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    datasets = data.get("datasets", [])
    print(f"  Found {len(datasets)} datasets in collection")
    for ds in datasets:
        title = ds.get("title", ds.get("name", "unknown"))
        ds_id = ds.get("dataset_id", ds.get("id", "unknown"))
        cell_count = ds.get("cell_count", "?")
        print(f"    - {title} (id={ds_id}, cells={cell_count})")
        # Find h5ad asset
        for asset in ds.get("assets", []):
            if asset.get("filetype") == "H5AD":
                print(f"      H5AD URL available (size: {asset.get('filesize', '?')} bytes)")
    return datasets

def download_h5ad_via_discover_api(collection_id: str, output_path: Path, 
                                     preferred_title: str = None) -> bool:
    """Download an h5ad file using the Discover API presigned URL."""
    try:
        datasets = get_collection_datasets_via_api(collection_id)
        if not datasets:
            print("  No datasets found via Discover API")
            return False
        
        # Pick the right dataset
        target = None
        for ds in datasets:
            title = ds.get("title", ds.get("name", ""))
            if preferred_title and preferred_title.lower() in title.lower():
                target = ds
                break
        if target is None:
            # Fall back to largest or first
            target = max(datasets, key=lambda d: d.get("cell_count", 0))
        
        # Find H5AD asset URL
        h5ad_url = None
        for asset in target.get("assets", []):
            if asset.get("filetype") == "H5AD":
                h5ad_url = asset.get("url")
                break
        
        if not h5ad_url:
            # Try the dataset download endpoint
            ds_id = target.get("dataset_id", target.get("id"))
            asset_url = f"https://api.cellxgene.cziscience.com/dp/v1/datasets/{ds_id}/assets"
            print(f"  Querying asset endpoint: {asset_url}")
            r = requests.get(asset_url, timeout=30)
            if r.ok:
                assets = r.json()
                for asset in assets if isinstance(assets, list) else assets.get("assets", []):
                    if asset.get("filetype") == "H5AD":
                        h5ad_url = asset.get("url", asset.get("presigned_url"))
                        break
        
        if not h5ad_url:
            print("  Could not find H5AD download URL via Discover API")
            return False
        
        print(f"  Downloading from Discover API (AWS S3 presigned URL)...")
        download_with_progress(h5ad_url, output_path)
        return True
        
    except Exception as e:
        print(f"  Discover API failed: {e}")
        return False

# ─── Strategy 2: CELLxGENE Census API ───────────────────────────────────────
def download_via_census(collection_id: str, output_path: Path) -> bool:
    """Download using the Census API (lists datasets, then downloads)."""
    try:
        import cellxgene_census
        
        print(f"  Opening Census (this may take a moment)...")
        with cellxgene_census.open_soma(census_version="2025-11-08") as census:
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()
            print(f"  Census has {len(datasets)} total datasets")
            print(f"  Columns: {list(datasets.columns)}")
            
            collection_ds = datasets[datasets["collection_id"] == collection_id]
            print(f"  Found {len(collection_ds)} datasets for collection {collection_id}")
            
            if collection_ds.empty:
                print("  No datasets found in Census for this collection")
                return False
            
            print(collection_ds[["dataset_id", "dataset_title"]].to_string())
            
            # Pick the first/largest
            target = collection_ds.iloc[0]
            dataset_id = target["dataset_id"]
            print(f"  Downloading dataset_id={dataset_id}...")
            
            cellxgene_census.download_source_h5ad(
                dataset_id=dataset_id, 
                to_path=str(output_path),
                census_version="2025-11-08"
            )
            return True
            
    except Exception as e:
        print(f"  Census download failed: {e}")
        return False

# ─── Shared download utility ────────────────────────────────────────────────
def download_with_progress(url: str, dest: Path, chunk_size: int = 65536):
    """Download with progress bar using larger chunks for speed."""
    from tqdm import tqdm
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(
            desc=dest.name, total=total,
            unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)
    print(f"  ✓ Downloaded to {dest}")

# ─── Main orchestrator ──────────────────────────────────────────────────────
def main():
    config = load_config()
    datasets = config.get("datasets", {})
    
    for key, ds_info in datasets.items():
        final_path = Path(ds_info["local_path"])
        temp_path = final_path.with_suffix('.h5ad.new')  # Download to temp file
        collection_id = ds_info.get("cellxgene_collection_id")
        
        # Skip if we already have a valid (complete) file
        # HNOCA from Zenodo is 18.7 GB; if our file is close to that, it might be complete
        if final_path.exists():
            size_gb = final_path.stat().st_size / 1e9
            if key == "hnoca" and size_gb < 17:
                print(f"\n{'='*60}")
                print(f"  {key}: Partial Zenodo download detected ({size_gb:.1f} GB)")
                print(f"  Attempting faster download via CELLxGENE APIs...")
                print(f"  (Partial file preserved at {final_path})")
                print(f"{'='*60}")
            else:
                print(f"\n{key}: Already downloaded ({size_gb:.1f} GB). Skipping.")
                continue
        else:
            print(f"\n{'='*60}")
            print(f"  Downloading {key}: {ds_info['name']}")
            print(f"{'='*60}")
        
        if not collection_id:
            print(f"  No collection_id for {key}, skipping")
            continue
        
        success = False
        
        # Strategy 1: Discover API (fastest — presigned S3 URLs)
        print(f"\n[Strategy 1] CELLxGENE Discover API...")
        success = download_h5ad_via_discover_api(collection_id, temp_path)
        
        # Strategy 2: Census API
        if not success:
            print(f"\n[Strategy 2] CELLxGENE Census API...")
            success = download_via_census(collection_id, temp_path)
        
        # If we got a new file, validate and swap
        if success and temp_path.exists():
            new_size = temp_path.stat().st_size
            print(f"  New file size: {new_size / 1e9:.2f} GB")
            
            if new_size > 1_000_000:  # At least 1 MB — sanity check
                if final_path.exists():
                    old_size = final_path.stat().st_size
                    backup = final_path.with_suffix('.h5ad.zenodo_partial')
                    print(f"  Backing up partial Zenodo download ({old_size/1e9:.1f} GB) → {backup.name}")
                    shutil.move(str(final_path), str(backup))
                
                shutil.move(str(temp_path), str(final_path))
                print(f"  ✓ {key} ready at {final_path}")
            else:
                print(f"  ✗ Downloaded file too small ({new_size} bytes), keeping existing")
                temp_path.unlink()
        elif not success:
            print(f"  ✗ All strategies failed for {key}")
            if temp_path.exists():
                temp_path.unlink()

    # Now run schema inspection
    print(f"\n{'='*60}")
    print("  Running schema inspection on downloaded files...")
    print(f"{'='*60}")
    
    sys.path.insert(0, '.')
    from src.data.inspect_obs import inspect_all, save_schemas, print_schema_report, load_config as load_config2
    
    cfg = load_config2()
    schemas = inspect_all()
    if schemas:
        out_dir = Path(cfg.get("outputs", {}).get("schemas", "results/pilot/schemas/"))
        save_schemas(schemas, out_dir)
        print_schema_report(schemas)
    else:
        print("  No schemas to inspect (no complete downloads yet)")

if __name__ == '__main__':
    main()
