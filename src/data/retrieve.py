import yaml
from pathlib import Path
import cellxgene_census
import requests
import hashlib
from tqdm import tqdm
import pandas as pd

def load_config(config_path: str = 'config/paths.yaml') -> dict:
    """Loads path configuration from yaml."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_sha256(file_path: Path) -> str:
    """Computes SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

import time

def download_file(url: str, dest: Path, chunk_size: int = 8192, max_retries: int = 1000) -> Path:
    """Downloads a file with a progress bar, retries, and resume capability."""
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
                
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code == 416: # Range not satisfiable (already downloaded)
                    print(f"Already fully downloaded (416): {dest}")
                    return dest
                    
                r.raise_for_status()
                
                # If server ignores Range header, it will return 200 instead of 206
                if r.status_code == 200 and initial_pos > 0:
                    mode = 'wb'
                    initial_pos = 0
                    
                total_length = int(r.headers.get('content-length', 0)) + initial_pos
                
                with open(dest, mode) as f, tqdm(
                    desc=dest.name,
                    total=total_length,
                    initial=initial_pos,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)
            print(f"Downloaded to {dest}")
            return dest
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}. Retrying {attempt+1}/{max_retries}...")
            time.sleep(3)
            
    raise Exception(f"Failed to download {url} after {max_retries} attempts.")

def discover_datasets(collection_id: str) -> pd.DataFrame:
    """Uses cellxgene_census to list datasets in a collection."""
    with cellxgene_census.open_soma() as census:
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()
        collection_datasets = datasets[datasets["collection_id"] == collection_id]
        return collection_datasets[["dataset_id", "dataset_title"]]

def download_from_census(dataset_id: str, output_path: Path) -> Path:
    """Downloads specific .h5ad via census API."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cellxgene_census.download_source_h5ad(dataset_id=dataset_id, to_path=str(output_path))
    print(f"Downloaded dataset {dataset_id} to {output_path}")
    return output_path

def download_from_url(url: str, output_path: Path) -> Path:
    """Direct URL download (for Zenodo/direct links)."""
    return download_file(url, output_path)

def retrieve_all(config_path: str = 'config/paths.yaml') -> dict[str, Path]:
    """Orchestrator: downloads all three datasets."""
    config = load_config(config_path)
    datasets = config.get("datasets", {})
    output_paths = {}

    for key, ds_info in datasets.items():
        local_path = Path(ds_info["local_path"])

        print(f"Retrieving {key}: {ds_info['name']}")
        
        # Prefer Zenodo URL if available, else try census
        if "zenodo_url" in ds_info:
            download_from_url(ds_info["zenodo_url"], local_path)
        elif "cellxgene_collection_id" in ds_info:
            if local_path.exists() and local_path.stat().st_size > 1000000:
                print(f"{key} already seems downloaded from census. Skipping.")
            else:
                print(f"Querying census for collection {ds_info['cellxgene_collection_id']}")
                try:
                    df = discover_datasets(ds_info["cellxgene_collection_id"])
                    print("Available datasets in collection:")
                    print(df.columns)
                    if not df.empty:
                        # Grab the first dataset
                        first_ds = df.iloc[0]
                        dataset_id = first_ds['dataset_id']
                        print(f"Downloading dataset {dataset_id} from collection.")
                        download_from_census(dataset_id, local_path)
                    else:
                        print(f"No datasets found for collection {ds_info['cellxgene_collection_id']}")
                except Exception as e:
                    print(f"Error querying census for {key}: {e}")
                
        if local_path.exists():
            try:
                checksum = compute_sha256(local_path)
                print(f"SHA256 for {key}: {checksum}")
                output_paths[key] = local_path
            except Exception as e:
                print(f"Error computing checksum for {key}: {e}")

    return output_paths

if __name__ == '__main__':
    retrieve_all()
