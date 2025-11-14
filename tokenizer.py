import json
from datasets import load_from_disk
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def tokenize_dataset(dataset, max_length=64, padding=True):
    with open("vocab_dict.json", "r") as f:
        vocab_dict = json.load(f)
    training_data = []
    validation_data = []
    for data_subset in ["train", "validation"]:
        for sequence in tqdm(dataset[data_subset]):
            tokens = sequence["tokens"]
            input_ids_tokens = (
                [vocab_dict["<bos>"]]
                + [
                    vocab_dict[x] if x in vocab_dict else vocab_dict["<unk>"]
                    for x in tokens
                ]
                + [vocab_dict["<eos>"]]
            )

            input_ids = []
            for i in range(0, len(input_ids_tokens), max_length):
                start_index = i
                end_index = min(start_index + max_length + 1, len(input_ids_tokens))
                batch = input_ids_tokens[start_index:end_index]
                batch = batch + [vocab_dict["<pad>"]] * (max_length + 1 - len(batch))
                input_ids.append(batch)

            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            if len(input_ids.shape) == 0 or len(input_ids.shape) == 1:
                continue
            batch, _ = input_ids.shape
            for batch_idx in range(batch):
                if data_subset == "train":
                    training_data.append(
                        {
                            "input_ids": input_ids[batch_idx],
                        }
                    )
                elif data_subset == "validation":
                    validation_data.append(
                        {
                            "input_ids": input_ids[batch_idx],
                        }
                    )

    train_dataset = Dataset.from_list(training_data)
    validation_dataset = Dataset.from_list(validation_data)
    datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    datasets.save_to_disk(f"TinyStories_tokenized_{max_length}")


if __name__ == "__main__":
    dataset = load_from_disk("TinyStories_processed")
    tokenize_dataset(dataset, max_length=64, padding=True)
