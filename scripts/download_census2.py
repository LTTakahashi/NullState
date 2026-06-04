import sys
from pathlib import Path
import yaml

sys.path.append('src')
from data.retrieve import retrieve_all, load_config

config = load_config('config/paths.yaml')
if 'hnoca' in config['datasets']:
    del config['datasets']['hnoca']
    with open('config/paths.yaml', 'w') as f:
        yaml.dump(config, f)

print("Starting fixed census downloads...")
retrieve_all()
print("Finished.")
