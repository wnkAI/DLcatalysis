#!/bin/bash
# Self-contained GROVER fingerprint pipeline.
# Reads final_smi.csv -> grover_substrates.csv -> GROVER fingerprint LMDB.
# Run from DataSet/brenda2/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_DATA_DIR="$SCRIPT_DIR/../final_data"

# Step A: Generate grover_substrates.csv from final_smi.csv
echo "[GROVER] Generating grover_substrates.csv from final_smi.csv ..."
python -c "
import pandas as pd
df = pd.read_csv('$SCRIPT_DIR/final_smi.csv').dropna(subset=['SMILES'])
out = df[['SMILES']].rename(columns={'SMILES': 'substrates'})
out.to_csv('$FINAL_DATA_DIR/grover_substrates.csv', index=False)
print(f'  Wrote {len(out)} substrates -> $FINAL_DATA_DIR/grover_substrates.csv')
"

# Step B: GROVER feature extraction
cd ../../other_softwares/grover_software
export PYTHONPATH=$PYTHONPATH:$(pwd)
python scripts/save_features.py --data_path ../../DataSet/final_data/grover_substrates.csv --save_path ../../DataSet/final_data/grover_substrates.npz --features_generator fgtasklabel --restart
python main.py fingerprint --data_path ../../DataSet/final_data/grover_substrates.csv --features_path ../../DataSet/final_data/grover_substrates.npz --checkpoint_path ../../DataSet/checkpoint/grover_large.pt --fingerprint_source both --output ../../DataSet/final_data/grover_fingerprint.npz --save_lmdb_path ../../DataSet/final_data/grover_fingerprint.lmdb
