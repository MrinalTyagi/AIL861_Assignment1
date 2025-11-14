import fasttext
import json
from tqdm import tqdm
import torch

fasttext_model = fasttext.load_model("cc.en.300.bin")
vocab = json.load(open("vocab_dict.json"))
embedding_matrix = None
special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
input_dim = 300
for word, idx in tqdm(vocab.items()):
    if word not in special_tokens:
        embed = torch.tensor(fasttext_model.get_word_vector(word)).unsqueeze(0)
        if embedding_matrix is None:
            embedding_matrix = embed
        else:
            embedding_matrix = torch.cat([embedding_matrix, embed], dim=0)
embedding_matrix = torch.cat(
    [torch.empty(size=(len(special_tokens), input_dim)).normal_(mean=0, std=0.02), embedding_matrix], dim=0
)
print("Embedding save of shape: ", embedding_matrix.shape)
torch.save(embedding_matrix, "fasttext_embedding.pt")
