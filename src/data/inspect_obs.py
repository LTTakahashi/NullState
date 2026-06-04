import json
from pathlib import Path
import anndata as ad
from rich.console import Console
from rich.table import Table
import yaml

def load_config(config_path: str = 'config/paths.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def inspect_h5ad(path: Path, backed: bool = True) -> dict:
    """Loads .h5ad and returns a dict with schema information."""
    print(f"Inspecting {path}...")
    try:
        # Load in backed mode to save RAM
        adata = ad.read_h5ad(path, backed='r' if backed else None)
        
        obs_df = adata.obs
        obs_columns = list(obs_df.columns)
        obs_dtypes = {col: str(dtype) for col, dtype in obs_df.dtypes.items()}
        
        obs_categories = {}
        for col in obs_df.select_dtypes(include=['category', 'object']).columns:
            uniques = obs_df[col].unique()
            # Cap at 50 to avoid massive JSONs for things like barcodes
            if len(uniques) < 50:
                obs_categories[col] = [str(x) for x in uniques]
            else:
                obs_categories[col] = f"<Too many unique values: {len(uniques)}>"

        schema = {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "obs_columns": obs_columns,
            "obs_dtypes": obs_dtypes,
            "obs_categories": obs_categories,
            "layers": list(adata.layers.keys()),
            "obsm_keys": list(adata.obsm.keys()),
            "uns_keys": list(adata.uns.keys()),
        }
        
        if backed:
            adata.file.close()
            
        return schema
    except Exception as e:
        print(f"Error inspecting {path}: {e}")
        return {}

def inspect_all(config_path: str = 'config/paths.yaml') -> dict[str, dict]:
    """Inspect all datasets."""
    config = load_config(config_path)
    datasets = config.get("datasets", {})
    schemas = {}
    
    for key, ds_info in datasets.items():
        local_path = Path(ds_info["local_path"])
        if local_path.exists():
            schemas[key] = inspect_h5ad(local_path)
        else:
            print(f"Warning: {local_path} does not exist. Run retrieve.py first.")
            
    return schemas

def save_schemas(schemas: dict, output_dir: Path):
    """Save schemas as JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, schema in schemas.items():
        out_file = output_dir / f"{key}_schema.json"
        with open(out_file, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Saved schema to {out_file}")

def print_schema_report(schemas: dict):
    """Pretty-print using rich."""
    console = Console()
    for key, schema in schemas.items():
        if not schema:
            continue
            
        console.print(f"\n[bold green]Dataset: {key}[/bold green]")
        console.print(f"Cells: [bold]{schema['n_obs']}[/bold] | Genes: [bold]{schema['n_vars']}[/bold]")
        
        table = Table(title=f".obs Columns ({len(schema['obs_columns'])})")
        table.add_column("Column Name", style="cyan")
        table.add_column("Dtype", style="magenta")
        table.add_column("Sample Values / Categories", style="yellow")
        
        for col in schema['obs_columns']:
            dtype = schema['obs_dtypes'][col]
            sample = schema['obs_categories'].get(col, "Continuous/Many")
            if isinstance(sample, list):
                sample = ", ".join(sample[:5]) + ("..." if len(sample) > 5 else "")
            table.add_row(col, dtype, str(sample))
            
        console.print(table)
        console.print(f"Layers: {', '.join(schema['layers']) if schema['layers'] else 'None'}")
        console.print(f"obsm keys: {', '.join(schema['obsm_keys']) if schema['obsm_keys'] else 'None'}")

if __name__ == '__main__':
    config = load_config()
    schemas = inspect_all()
    out_dir = Path(config.get("outputs", {}).get("schemas", "results/pilot/schemas/"))
    save_schemas(schemas, out_dir)
    print_schema_report(schemas)
