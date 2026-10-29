#!/usr/bin/env bash
# Set up torch_xla on a Colab TPU runtime (or TPU VM) and run the full DFlash Pareto sweep.
#
#   bash scripts/tpu/run_sweep.sh            # setup + equivalence test + AR + DFlash sweeps + plot
#   SKIP_SETUP=1 bash scripts/tpu/run_sweep.sh
#
# Results: docs/data/pareto_tpu.jsonl, logs in docs/data/sweep_tpu.log, plot docs/pareto_tpu.png.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SKIP_SETUP:-}" ]; then
  # torch_xla pins its matching torch; transformers/tokenizers match the repo's pin (5.15 still
  # accepts the positive DynamicCache.crop that dflash.batched uses).
  pip install -q "torch_xla[tpu]" -f https://storage.googleapis.com/libtpu-releases/index.html
  pip install -q "transformers==5.15.0" "tokenizers==0.22.2" "datasets" "matplotlib" "accelerate"
  pip install -q -e . --no-deps
fi
python -c "import torch, torch_xla, transformers; print('torch', torch.__version__, 'torch_xla', torch_xla.__version__, 'transformers', transformers.__version__); print(torch_xla.device())"

export PYTHONPATH=.:${PYTHONPATH:-}
LOG=docs/data/sweep_tpu.log
mkdir -p docs/data

echo "== equivalence test on the TPU (tiny random models; device xla) ==" | tee -a $LOG
python scripts/hetero/test_batched_xla.py xla 2>&1 | grep -v -i warning | tee -a $LOG

BATCHES="${BATCHES:-1 2 4 8 16 32 64 128}"
BLOCKS="${BLOCKS:-4 8 16}"
python scripts/tpu/pareto_tpu.py --system ar --batch $BATCHES 2>&1 | tee -a $LOG
python scripts/tpu/pareto_tpu.py --system dflash --block $BLOCKS --batch $BATCHES 2>&1 | tee -a $LOG
python scripts/tpu/plot_pareto_tpu.py docs/data/pareto_tpu.jsonl docs/pareto_tpu.png --with-ryzen docs/data/pareto.jsonl
