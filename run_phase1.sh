#!/bin/bash
cd "$(dirname "$0")"

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements for Phase 1
pip install -r requirements.txt

# Run Phase 1
export PYTHONPATH=.
python src/data/retrieve.py
python src/data/inspect_obs.py
