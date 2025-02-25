from datasets import DatasetDict
import random

random.seed(42)


def get_random_train_subset(dataset: DatasetDict, train_size: int) -> DatasetDict:

    train_split = dataset["train"]
    test_split = dataset["test"]

    random_subset = random.sample(range(len(train_split)), train_size)
    train_split = train_split.select(random_subset)

    sampled_dataset = DatasetDict()

    sampled_dataset["train"] = train_split
    sampled_dataset["test"] = test_split

    return sampled_dataset
