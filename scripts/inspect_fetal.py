import scanpy as sc
adata = sc.read_h5ad('data/cao_fetal.h5ad', backed='r')
print('Columns:', list(adata.obs.columns))
if 'cell_type' in adata.obs.columns:
    print('Cell types sample:', adata.obs['cell_type'].unique()[:50].tolist())
if 'tissue' in adata.obs.columns:
    print('Tissues:', adata.obs['tissue'].unique().tolist())
