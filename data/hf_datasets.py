from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets
from typing import Dict, List

KEY = "content"

def filter_code(code: str) -> bool:
    return "cvxpy" in code

def filter_example(example: Dict[str, str]) -> bool:
    return filter_code(example[KEY])

def get_streaming_dataset() -> IterableDataset:
    datasets = []

    # The stack
    dataset = load_dataset('bigcode/the-stack', data_dir='data/python', streaming=True)
    dataset = interleave_datasets(dataset.values())
    dataset = dataset.rename_column('content', KEY) if KEY != 'content' else dataset
    column_names_to_remove = [key for key in dataset.column_names if key != KEY]
    dataset = dataset.remove_columns(column_names_to_remove)
    datasets.append(dataset)

    # Github Code
    dataset = load_dataset('codeparrot/github-code', streaming=True, languages=["Python"], trust_remote_code=True)
    dataset = interleave_datasets(dataset.values())
    dataset = dataset.rename_column('code', KEY) if KEY != 'code' else dataset
    column_names_to_remove = [key for key in dataset.column_names if key != KEY]
    dataset = dataset.remove_columns(column_names_to_remove)
    datasets.append(dataset)

    # Interleave datasets
    dataset = interleave_datasets(datasets)

    # Fiter dataset
    dataset = dataset.filter(filter_example, batched=False)
    return dataset

def save_streaming_dataset(dataset: IterableDataset, dataset_path: str):
    print("Converting from IterableDataset to Dataset...")
    new_dataset = Dataset.from_generator(lambda: dataset, num_proc=8)
    print("Saving dataset to disk...")
    new_dataset.save_to_disk(dataset_path=dataset_path, max_shard_size="500MB", num_proc=8)

def main():
    streaming_dataset = get_streaming_dataset()
    save_streaming_dataset(streaming_dataset, "dataset_hf_datasets_python_cvxpy")

if __name__ == '__main__':
    main()
