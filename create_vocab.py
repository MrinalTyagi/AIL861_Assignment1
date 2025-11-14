from datasets import load_from_disk
from tqdm import tqdm
import json
from collections import Counter


def create_vocab(dataset):
    vocab_list = []
    for subset in ["train", "validation"]:
        for element in tqdm(dataset[subset]):
            tokens = element["tokens"]
            for token in tokens:
                vocab_list.append(token)
    return vocab_list

max_vocab_size = 30000

dataset = load_from_disk("TinyStories_processed")
vocab_list = create_vocab(dataset)
counter_vocab_dict = Counter(vocab_list)
counter_vocab = counter_vocab_dict.most_common(max_vocab_size - 4)
vocab_list = ["<pad>", "<unk>", "<bos>", "<eos>"] + [x[0] for x in counter_vocab]


final_vocab_dict = {vocab_list[i]: i for i in range(len(vocab_list))}
inverse_vocab_dict = {v: k for k, v in final_vocab_dict.items()}
with open("vocab_dict.json", "w") as f:
    json.dump(final_vocab_dict, f)

inverse_vocab_dict = {v: k for k, v in final_vocab_dict.items()}
with open("inverse_vocab_dict.json", "w") as f:
    json.dump(inverse_vocab_dict, f)
