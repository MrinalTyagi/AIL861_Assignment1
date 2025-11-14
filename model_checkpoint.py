import torch
from torch import nn
import numpy as np

eps = 1e-9

class GradientCheckpointing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *inputs):
        ctx.module = module
        ctx.save_for_backward(*inputs)
        with torch.no_grad():
            nograd_output = module(*inputs)
        return nograd_output
    
    @staticmethod
    def backward(ctx, grad_output):
        layer = ctx.module
        ctx_inputs = ctx.saved_tensors
        with torch.enable_grad():
            layer_output = layer(*ctx_inputs)
        ctx_inputs = [x.requires_grad_(True) for x in ctx_inputs]
        grads = torch.autograd.grad(
            layer_output,
            ctx_inputs + list(layer.parameters()),
            grad_outputs=grad_output,
            allow_unused=True
        )
        input_gradients = grads[:len(ctx_inputs)]
        parameter_gradients = grads[len(ctx_inputs):]
        for parameter, gradient in zip(layer.parameters(), parameter_gradients):
            parameter.grad = gradient
        return None, *input_gradients

class GradientCheckpointingModule(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *inputs):
        return GradientCheckpointing.apply(self.layer, *inputs)


def attention(query, key, value, mask=None):
    d_k = torch.tensor(query.shape[-1], device=query.device)
    attn_score = query @ key.transpose(-1, -2) / torch.sqrt(d_k)
    causal_mask = torch.triu(
        torch.ones(attn_score.shape, device=attn_score.device), diagonal=1
    )
    final_mask = causal_mask.bool()
    if mask is not None:
        final_mask = final_mask.masked_fill(
            mask.unsqueeze(1).unsqueeze(2).bool() == True, 1
        )
    attn_score = attn_score.masked_fill(final_mask.float() == 1, float("-inf"))
    softmax_score = nn.functional.softmax(attn_score, dim=-1)
    softmax_score = softmax_score.masked_fill(torch.isnan(softmax_score), 0)
    output = softmax_score @ value
    return output


class PositionEmbedding(nn.Module):
    def __init__(self, context_size, dmodel):
        super(PositionEmbedding, self).__init__()
        self.context_size = context_size
        self.dmodel = dmodel

        table = np.zeros(shape=(self.context_size, self.dmodel))
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if j % 2 == 0:
                    table[i, j] = np.sin(i / np.power(500., (j) / self.dmodel))
                else:
                    table[i, j] = np.cos(i / np.power(500., (j) / self.dmodel))
        self.positional_embedding_table = torch.tensor(table, requires_grad=False, device="cuda", dtype=torch.float32).unsqueeze(0)

    def forward(self, inputs):
        batch_size, seq_length, hidden_dim = inputs.shape
        return self.positional_embedding_table[:, :seq_length, :] + inputs


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        input_dim,
        hidden_dim,
        context_size,
        vocab_size,
        vocab,
        checkpoint=True
    ):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.fasttext_embedding = nn.Embedding(self.vocab_size, self.input_dim)
        self.pos_embed = PositionEmbedding(self.context_size, self.input_dim)
        self.special_token_embedding = nn.Embedding(4, self.input_dim)
        self.fasttext_embedding.weight = nn.Parameter(
            torch.load("fasttext_embedding.pt"), requires_grad=False
        )
        print(
            "Embedding weight shape: ",
            self.fasttext_embedding.weight.shape,
            " and sum: ",
            self.fasttext_embedding.weight.sum(),
        )
        self.transformer_layers = nn.ModuleList(
            [
                GradientCheckpointingModule(TransformerBlock(self.num_heads, self.input_dim, self.hidden_dim)) if checkpoint else TransformerBlock(self.num_heads, self.input_dim, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.ln = GradientCheckpointingModule(LayerNorm(self.input_dim)) if checkpoint else LayerNorm(self.input_dim)
        self.linear = GradientCheckpointingModule(nn.Linear(self.input_dim, self.vocab_size)) if checkpoint else nn.Linear(self.input_dim, self.vocab_size)

    def forward(self, inputs, checkpoint=True):
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"].type(torch.float32)


        embedded_output = self.fasttext_embedding(input_ids)
        special_token_embedded_output = self.special_token_embedding(torch.clamp(input_ids, max=3))
        mask = (input_ids < 4)
        embedded_output[mask] = special_token_embedded_output[mask]
        pos_embedded_output = self.pos_embed(embedded_output)
        mhsa_output = pos_embedded_output
        for layer in self.transformer_layers:
            mhsa_output = layer(mhsa_output, attn_mask)
        output = self.linear(self.ln(mhsa_output))
        return output


class LayerNorm(nn.Module):
    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.input_dim = input_dim
        self.scale = nn.Parameter(torch.randn(self.input_dim))
        self.shift = nn.Parameter(torch.randn(self.input_dim))

    def forward(self, inputs):
        mu = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, unbiased=True, keepdim=True)
        x_bar = (inputs - mu) / (std + eps)
        output = self.scale * x_bar + self.shift
        return output


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mhsa = MultiHeadSelfAttention(
            self.num_heads, self.input_dim, self.hidden_dim
        )
        self.ln1 = LayerNorm(self.input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, 4 * self.input_dim),
            nn.ReLU(),
            nn.Linear(4 * self.input_dim, self.input_dim),
        )
        self.ln2 = LayerNorm(self.input_dim)

    def forward(self, inputs, attn_mask=None):
        output1 = self.ln1(inputs)
        output1 = self.mhsa(output1, attn_mask) + inputs
        output2 = self.ln2(output1)
        output2 = self.ffn(output2) + output2
        return output2


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Wk = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wq = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wv = nn.Linear(self.input_dim, self.hidden_dim)

        self.Wo = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, inputs, attn_mask=None):
        batch_size, seq_length, hidden_dim = inputs.shape
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)

        q = q.view(
            batch_size, seq_length, self.num_heads, self.hidden_dim // self.num_heads
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_length, self.num_heads, self.hidden_dim // self.num_heads
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_length, self.num_heads, self.hidden_dim // self.num_heads
        ).transpose(1, 2)
        attn_output = attention(q, k, v, attn_mask).transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, seq_length, -1)
        output = self.Wo(attn_output)
        return output


if __name__ == "__main__":
    pass