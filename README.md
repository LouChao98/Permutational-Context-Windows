# Permutational Context Window

An experimental repo. Forked from AI21's Parallel Context Window, see `README_PCW.md` for details.

## Setup

As PCW

## Evaluation


### Naive

```bash
python run_evaluation.py \
--dataset sst2 \
--model openlm-research/open_llama_3b \
--n-windows 1 \
--n-shots-per-window 10 \
--subsample-test-set 250 \
--n-runs 30 \
--output-dir logs/sst2_naive
```

sbatch submit24.sh run_evaluation.py --dataset sst2 --model data/llama-7b --n-windows 1 --n-shots-per-window 10 --subsample-test-set 250 --n-runs 30 --output-dir logs/sst2_llama7b_naive

### PCW

```bash
python run_evaluation.py \
--dataset sst2 \
--model openlm-research/open_llama_3b \
--n-windows 2 \
--n-shots-per-window 5 \
--subsample-test-set 250 \
--n-runs 30 \
--output-dir logs/sst2_pcw
```

sbatch submit24.sh run_evaluation.py --dataset sst2 --model data/llama-7b --n-windows 2 --n-shots-per-window 5 --subsample-test-set 250 --n-runs 30 --output-dir logs/sst2_llama7b_pcw

### PermCW

```bash
python run_permcw_evaluation.py \
--dataset sst2 \
--model openlm-research/open_llama_3b \
--n-windows 10 \
--n-shots-per-window 1 \
--subsample-test-set 250 \
--n-runs 30 \
--output-dir logs/sst2_permcw
```

sbatch submit24.sh run_permcw_evaluation.py --dataset sst2 --model data/llama-7b --n-windows 10 --n-shots-per-window 1 --subsample-test-set 250 --n-runs 30 --output-dir logs/sst2_llama7b_permcw


## Results

See `logs`. Currently, only exp on `sst2` is available.