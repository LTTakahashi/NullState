import cellxgene_census

with cellxgene_census.open_soma() as c:
    df = c['census_info']['datasets'].read().concat().to_pandas()
    hdbca = df[df['collection_id'] == '4d8fed08-2d6d-4692-b5ea-464f1d072077']
    print(hdbca[['dataset_id', 'dataset_title', 'dataset_total_cell_count']].to_string())
