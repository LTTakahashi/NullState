import sys
import os
from pathlib import Path

# Add src to python path
sys.path.append('src')

from data.retrieve import download_from_census

print("Starting census downloads...")
download_from_census('4d8fed08-2d6d-4692-b5ea-464f1d072077', Path('data/hdbca.h5ad'))
download_from_census('c114c20f-1ef4-49a5-9c2e-d965787fb90c', Path('data/cao_fetal.h5ad'))
print("Finished census downloads.")
