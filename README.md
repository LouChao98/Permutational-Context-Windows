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

Results:
```plain
accuracy = 0.684
accuracy = 0.72
accuracy = 0.66
accuracy = 0.716
accuracy = 0.628
accuracy = 0.7
accuracy = 0.708
accuracy = 0.68
accuracy = 0.656
accuracy = 0.688
accuracy = 0.616
accuracy = 0.716
...
```

Average: 0.6861


### PermCW

```bash
python run_permcw_evaluation.py \
--dataset sst2 \
--model openlm-research/open_llama_3b \
--n-windows 2 \
--n-shots-per-window 5 \
--subsample-test-set 250 \
--n-runs 30 \
--output-dir logs/sst2_permcw
```