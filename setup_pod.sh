#!/bash/bin
# Setup script for Parameter Golf pod

echo "--- Installing dependencies ---"
pip install -r requirements.txt

echo "--- Downloading tokenizer and dataset shards ---"
# Defaulting to sp1024 variant with 1 shard for warmup
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

echo "--- Setup Complete ---"
