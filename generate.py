import os
from datasets import load_from_disk, load_dataset
import argparse
from model import Decoder
import torch
import json
from torch import nn
from tqdm import tqdm
import numpy as np
import evaluate
import time


class Generate:
    def __init__(self, checkpoint_path, temperature=1, decoding="greey", beam_width=5, experiment_name="default"):
        self.experiment_name = experiment_name
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path)
        self.config = self.checkpoint["config"]
        self.decoding = decoding
        self.temperature = temperature
        self.beam_width = beam_width
        self.vocab_dict = json.load(open(self.config["dataset"]["vocab_file"]))
        self.inverse_vocab_dict = json.load(
            open(self.config["dataset"]["inverse_vocab_file"])
        )
        model_kwargs = {
            "num_layers": self.config["model"]["num_layers"],
            "num_heads": self.config["model"]["num_heads"],
            "input_dim": self.config["model"]["input_dim"],
            "hidden_dim": self.config["model"]["hidden_dim"],
            "context_size": self.config["model"]["context_size"],
            "vocab_size": len(self.vocab_dict),
            "vocab": self.vocab_dict,
        }
        self.model = Decoder(**model_kwargs)
        self.bleu_calculator = evaluate.load("bleu")
        checkpoint = self.checkpoint["model"]
        updated_checkpoint = {}
        for key, value in checkpoint.items():
            updated_checkpoint[key.replace("module.", "")] = value
        self.model.load_state_dict(updated_checkpoint)
        self.model.eval()
        self.model.to("cuda")
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.vocab_dict["<pad>"]
        )
        self.validation_dataset = load_from_disk(
            self.config["dataset"]["original_dataset"]
        )["validation"]

        self.pretrain_dataset = load_from_disk(self.config["dataset"]["name"])
        self.validation_dataloader = torch.utils.data.DataLoader(
            self.pretrain_dataset["validation"],
            batch_size=self.config["validation"]["batch_size"],
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    def collate_fn(self, inputs):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in inputs])
        return {
            "input_ids": input_ids,
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_validation_loss = 0.0
        validation_progress_bar = tqdm(
            self.validation_dataloader,
            desc="Validating",
        )
        for data in validation_progress_bar:
            data["input_ids"] = data["input_ids"].to("cuda")
            input_ids = data["input_ids"][:, :-1].clone()
            attention_mask = (
                (input_ids == self.vocab_dict["<pad>"]).type(torch.int64).to("cuda")
            )
            labels = data["input_ids"][:, 1:].clone()
            outputs = self.model(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            _, _, vocab_size = outputs.shape
            validation_loss = self.loss_fn(
                outputs.view(-1, vocab_size), labels.view(-1)
            )
            total_validation_loss += validation_loss.item()
            validation_progress_bar.set_postfix(loss=validation_loss.item())

        validation_loss = total_validation_loss / len(self.validation_dataloader)
        print(f"Validation Loss: {validation_loss}")
        print(f"Validation Perplexity: {np.exp(validation_loss)}")
        print("-" * 100)
        return np.exp(validation_loss)

    def generate_one_sample_beam_search(self, sample):
        total_tokens = 0
        start_time = time.time()
        input_tokens = ["<bos>"] + sample["tokens"][:5]
        generated_tokens = []
        beam_scores = torch.zeros(self.beam_width).to("cuda")
        labels = sample["tokens"][
            len(input_tokens) : self.config["model"]["context_size"]
        ]
        labels = torch.tensor(
            [
                (
                    self.vocab_dict[x]
                    if x in self.vocab_dict
                    else self.vocab_dict["<unk>"]
                )
                for x in labels
            ],
            dtype=torch.int64,
        ).to("cuda")

        input_ids = (
            torch.tensor(
                [self.vocab_dict[x] if x in self.vocab_dict else self.vocab_dict["<unk>"] for x in input_tokens + generated_tokens],
                dtype=torch.int64,
            ).unsqueeze(0)
            .to("cuda")
        )
        for _ in range(self.config["model"]["context_size"] - len(input_tokens)):
            attention_mask = (
                (input_ids == self.vocab_dict["<pad>"]).type(torch.int64).to("cuda")
            )
            outputs = self.model(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            total_tokens += input_ids.flatten().shape[0]
            last_output = outputs[:, -1, :]
            log_probs = torch.log_softmax(last_output, dim=-1)
            log_probs_topk, indices_topk = log_probs.topk(self.beam_width, dim=-1)
            beam_scores = beam_scores.unsqueeze(0) + log_probs_topk
            beam_scores = beam_scores.view(-1)
            batch_size = input_ids.shape[0]
            input_ids = torch.cat([torch.repeat_interleave(input_ids, self.beam_width, dim=0), indices_topk.view(-1, 1)], dim=1)
            best_beam_scores, best_beam_indices = beam_scores.topk(self.beam_width, dim=0)
            input_ids = input_ids[best_beam_indices]
            beam_scores = best_beam_scores

        best_generated = input_ids[beam_scores.argmax()]
        prediction = " ".join(
            [self.inverse_vocab_dict[str(x)] for x in best_generated.detach().cpu().tolist()]
        )
        gt = " ".join(
            input_tokens + [self.inverse_vocab_dict[str(x)] for x in labels.detach().cpu().tolist()]
        )
        end_time = time.time()
        tokens_per_second = total_tokens / (end_time - start_time)
        print("Tokens per second: ", tokens_per_second)
        print("Prediction: ", prediction)
        print("GT: ", gt)
        print("--------------------------------")
        return prediction, gt, tokens_per_second

    def generate_one_sample(self, sample):
        total_tokens = 0
        start_time = time.time()
        input_tokens = ["<bos>"] + sample["tokens"][:5]
        generated_tokens = []
        labels = sample["tokens"][
            len(input_tokens) : self.config["model"]["context_size"]
        ]
        labels = torch.tensor(
            [
                (
                    self.vocab_dict[x]
                    if x in self.vocab_dict
                    else self.vocab_dict["<unk>"]
                )
                for x in labels
            ],
            dtype=torch.int64,
        ).to("cuda")
        while len(generated_tokens) < len(labels):
            input_ids = (
                torch.tensor(
                    [
                        (
                            self.vocab_dict[x]
                            if x in self.vocab_dict
                            else self.vocab_dict["<unk>"]
                        )
                        for x in input_tokens + generated_tokens
                    ],
                    dtype=torch.int64,
                )
                .unsqueeze(0)
                .to("cuda")
            )
            attention_mask = (
                (input_ids == self.vocab_dict["<pad>"]).type(torch.int64).to("cuda")
            )
            outputs = self.model(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            total_tokens += input_ids.flatten().shape[0]
            last_output = outputs[:, -1, :].squeeze(0)
            if self.decoding == "greedy":
                mapped_token = torch.argmax(torch.softmax(last_output / self.temperature, dim=-1)).item()
            elif self.decoding == "sampling": 
                mapped_token = torch.multinomial(
                    torch.softmax(last_output / self.temperature, dim=-1), num_samples=1
                ).item()
            else:
                raise ValueError(f"Invalid decoding method: {self.decoding}")
            generated_tokens.append(mapped_token)
            if mapped_token == self.vocab_dict["<eos>"]:
                break
        prediction = " ".join(
            input_tokens + [self.inverse_vocab_dict[str(x)] for x in generated_tokens]
        )
        gt = " ".join(
            input_tokens
            + [self.inverse_vocab_dict[str(x)] for x in labels.detach().cpu().tolist()]
        )
        end_time = time.time()
        tokens_per_second = total_tokens / (end_time - start_time)
        print("Tokens per second: ", tokens_per_second)
        print("Prediction: ", prediction)
        print("GT: ", gt)
        print("--------------------------------")
        return prediction, gt, tokens_per_second

    def generate(self, perplexity, num_samples=10):
        generated_sentences = []
        original_continuations = []
        tokens_per_seconds = []
        if num_samples != -1:
            ds = self.validation_dataset.select(range(num_samples))
        else:
            ds = self.validation_dataset
        start_time = time.time()
        for sample in tqdm(ds):
            if self.decoding == "beam_search":
                prediction, gt, tokens_per_second = self.generate_one_sample_beam_search(sample)
            elif self.decoding == "greedy" or self.decoding == "sampling":
                prediction, gt, tokens_per_second = self.generate_one_sample(sample)
            else:
                raise ValueError(f"Invalid decoding method: {self.decoding}")
            original_continuations.append(gt)
            generated_sentences.append(prediction)
            tokens_per_seconds.append(tokens_per_second)
        end_time = time.time()
        total_time = end_time - start_time

        bleu_results1 = self.bleu_calculator.compute(
            predictions=generated_sentences,
            references=original_continuations,
            max_order=1,
        )
        bleu_results2 = self.bleu_calculator.compute(
            predictions=generated_sentences,
            references=original_continuations,
            max_order=2,
        )

        print("BLEU Score 1: \n", bleu_results1["bleu"])
        print("BLEU Score 2: \n", bleu_results2["bleu"])
        os.makedirs(os.path.join(self.config["training"]["output_dir"], self.experiment_name), exist_ok=True)
        with open(
            os.path.join(self.config["training"]["output_dir"], self.experiment_name, "metrics.json"), "w"
        ) as f:
            json.dump(
                {
                    "bleu1": bleu_results1["bleu"],
                    "bleu2": bleu_results2["bleu"],
                    "validation_perplexity": perplexity,
                    "temperature": None if self.decoding == "beam_search" else self.temperature ,
                    "decoding": self.decoding,
                    "beam_width": self.beam_width if self.decoding == "beam_search" else None,
                    "num_samples": num_samples,
                    "tokens_per_seconds": np.mean(tokens_per_seconds),
                    "total_time": total_time
                },
                f,
            )

        responses = []
        for index, (original, generated) in enumerate(
            zip(original_continuations, generated_sentences)
        ):
            responses.append(
                {
                    "original": original,
                    "generated": generated,
                }
            )
        with open(
            os.path.join(self.config["training"]["output_dir"], self.experiment_name, "responses.json"), "w"
        ) as f:
            json.dump({"responses": responses}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=False, default=1)
    parser.add_argument("--beam_width", type=int, required=False, default=5)
    parser.add_argument("--decoding", type=str, required=False, default="sampling")
    parser.add_argument("--num_samples", type=int, required=False, default=10)
    parser.add_argument("--experiment_name", type=str, required=False, default="default")
    args = parser.parse_args()
    generate = Generate(args.checkpoint_path, args.temperature, args.decoding, args.beam_width, args.experiment_name)
    perplexity = generate.validate()
    generate.generate(perplexity, num_samples=args.num_samples)
