import sys
from pathlib import Path
sys.path.append('src')
from data.retrieve import download_from_census

print("Downloading correct full HDBCA dataset...")
download_from_census('13149914-ea03-4f01-8bf6-b793b667127b', Path('data/hdbca.h5ad'))
print("Finished downloading correct HDBCA dataset.")
