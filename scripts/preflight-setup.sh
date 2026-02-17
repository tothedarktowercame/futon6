#!/usr/bin/env bash
set -euo pipefail

# Preflight setup for Linode GPU (RTX 4000 Ada / Chicago)
# Run this once after provisioning. Takes ~10 min.

echo "=== 1. System packages ==="
apt-get update && apt-get install -y python3-pip python3-venv p7zip-full git

echo "=== 2. Clone and set up ==="
cd /root
git clone https://github.com/tothedarktowercame/futon6.git
cd futon6

echo "=== 3. Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Core deps + LWGM deps (torch with CUDA, faiss-cpu)
pip install -e ".[dev,lwgm]"

# Sentence transformers for BGE-large (Stage 2)
# accelerate for LLM device_map (Stages 3/6)
pip install sentence-transformers accelerate

echo "=== 4. Verify GPU ==="
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "=== 5. Quick test ==="
PYTHONPATH=src python3 -m pytest tests/test_graph_embed.py tests/test_faiss_index.py -q

echo "=== 6. Download math.SE data ==="
python3 scripts/superpod-job.py --download math --data-dir ./se-data

echo ""
echo "=== Setup complete ==="
echo "Now run the preflight:"
echo ""
echo "  cd /root/futon6 && source .venv/bin/activate"
echo ""
echo "  python3 scripts/superpod-job.py \\"
echo "      ./se-data/math.stackexchange.com/Posts.xml \\"
echo "      --comments-xml ./se-data/math.stackexchange.com/Comments.xml \\"
echo "      --site math.stackexchange \\"
echo "      --output-dir ./preflight-1000 \\"
echo "      --thread-limit 1000 \\"
echo "      --embed-device cuda \\"
echo "      --graph-embed-epochs 10 \\"
echo "      --preflight"
echo ""
echo "  python3 scripts/evaluate-superpod-run.py ./preflight-1000/ \\"
echo "      --json-report preflight-report.json"
