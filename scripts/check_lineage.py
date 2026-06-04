import scanpy as sc
adata = sc.read_h5ad('data/cao_fetal.h5ad', backed='r')
print('Lineages:', adata.obs['Organ_cell_lineage'].unique()[:50].tolist())
