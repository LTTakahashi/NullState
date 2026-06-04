import scanpy as sc
adata = sc.read_h5ad('data/cao_fetal.h5ad', backed='r')
print('Main_cluster_name:', adata.obs['Main_cluster_name'].unique().tolist())
