from datasets import load_dataset
from datasets_loader import DATASET_NAMES2LOADERS

for key, module in DATASET_NAMES2LOADERS.items():
    load_dataset(module.dataset).save_to_disk(f'data/{key}')