"""Validate the built reference.h5ad before committing GPU hours to training."""
import scanpy as sc
import numpy as np

print('Loading reference (backed mode)...')
ref = sc.read_h5ad('results/pilot/reference.h5ad', backed='r')
print(f'Shape: {ref.shape[0]:,} cells x {ref.shape[1]:,} genes')
print(f'obs columns: {list(ref.obs.columns)}')
print()

# Check ref_source distribution
if 'ref_source' in ref.obs.columns:
    print('=== ref_source counts ===')
    print(ref.obs['ref_source'].value_counts().to_string())
    print()

# Check cell_type distribution (top 30)
if 'cell_type' in ref.obs.columns:
    print('=== Top 30 cell types ===')
    print(ref.obs['cell_type'].value_counts().head(30).to_string())
    print()

# Check Main_cluster_name if present
if 'Main_cluster_name' in ref.obs.columns:
    print('=== Top 20 Main_cluster_name ===')
    print(ref.obs['Main_cluster_name'].value_counts().head(20).to_string())
    print()

# Check for NaNs in key columns
print('=== NaN check ===')
for col in ref.obs.columns:
    n_nan = ref.obs[col].isna().sum()
    if n_nan > 0:
        print(f'  {col}: {n_nan:,} NaN values')

# Check layers
print(f'\n=== Layers: {list(ref.layers.keys()) if ref.layers else "None"} ===')
print(f'obsm keys: {list(ref.obsm.keys()) if ref.obsm else "None"}')

# Check X value range (sample first 5 cells)
print('\n=== X matrix validation ===')
for i in range(min(5, ref.shape[0])):
    row = ref.X[i]
    if hasattr(row, 'toarray'):
        row = row.toarray().flatten()
    else:
        row = np.array(row).flatten()
    is_int = np.allclose(row[row != 0], np.round(row[row != 0]))
    print(f'  Cell {i}: min={row.min():.1f}, max={row.max():.1f}, '
          f'nonzero={np.count_nonzero(row)}/{len(row)}, integer_counts={is_int}')

# Summary
print('\n=== VALIDATION SUMMARY ===')
print(f'Total cells: {ref.shape[0]:,}')
print(f'Total genes: {ref.shape[1]:,}')
has_batch = 'batch' in ref.obs.columns
print(f'Has batch column: {has_batch}')
has_celltype = 'cell_type' in ref.obs.columns
print(f'Has cell_type column: {has_celltype}')
if has_celltype:
    n_types = ref.obs['cell_type'].nunique()
    print(f'Unique cell types: {n_types}')
print('REFERENCE LOOKS VALID' if ref.shape[0] > 100000 and ref.shape[1] > 10000 else 'WARNING: Reference may be too small')
