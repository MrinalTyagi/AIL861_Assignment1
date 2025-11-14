import os
from re import S
from datasets import load_from_disk, load_dataset
import argparse
import yaml
from model_checkpoint import Decoder
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch
import json
import random
import numpy as np
from tqdm import tqdm
import time
import torch.multiprocessing as mp
import wandb
import evaluate


class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        wandb.init(project="llm-pretraining", config=self.config)
        random.seed(self.config["training"]["seed"])
        np.random.seed(self.config["training"]["seed"])
        torch.manual_seed(self.config["training"]["seed"])
        torch.cuda.manual_seed(self.config["training"]["seed"])
        torch.cuda.manual_seed_all(self.config["training"]["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.bleu_calculator = evaluate.load("bleu")

        print("Loading vocabulary....")
        self.vocab_dict = json.load(open(self.config["dataset"]["vocab_file"]))
        self.inverse_vocab_dict = json.load(
            open(self.config["dataset"]["inverse_vocab_file"])
        )

        print("Loading pretraining dataset....")
        self.pretrain_dataset = load_from_disk(self.config["dataset"]["name"])
        print(
            "Pretraining Dataset with pretraining size ",
            len(self.pretrain_dataset["train"]),
            " and validation size ",
            len(self.pretrain_dataset["validation"]),
            " loaded.",
        )
        self.sampler = RandomSampler(
            self.pretrain_dataset["train"],
            replacement=False,
            num_samples=(
                self.config["training"]["num_samples"]
                if self.config["training"]["num_samples"] != -1
                else len(self.pretrain_dataset["train"])
            ),
        )

        print("Loading original dataset....")
        self.original_dataset = load_from_disk(
            self.config["dataset"]["original_dataset"]
        )

        print("Loading dataloaders...")
        self.pretrain_dataloader = DataLoader(
            self.pretrain_dataset["train"],
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            num_workers=16,
        )
        self.validation_dataloader = DataLoader(
            self.pretrain_dataset["validation"],
            batch_size=self.config["validation"]["batch_size"],
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

        print("Initializing model....")
        kwargs = {
            "num_layers": self.config["model"]["num_layers"],
            "num_heads": self.config["model"]["num_heads"],
            "input_dim": self.config["model"]["input_dim"],
            "hidden_dim": self.config["model"]["hidden_dim"],
            "context_size": self.config["model"]["context_size"],
            "vocab_size": len(self.vocab_dict),
            "vocab": self.vocab_dict,
            "checkpoint": True,
        }
        self.model = Decoder(**kwargs)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to("cuda")
        print(
            "Model successfully loaded with parameters ",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

        print("Initializing optimizer...")
        if self.config["training"]["optimizer"]["name"] == "adam":
            lr = float(self.config["training"]["optimizer"]["lr"])
            weight_decay = float(self.config["training"]["optimizer"]["weight_decay"])
            self.optimizer = torch.optim.Adam(
                {p for p in self.model.parameters() if p.requires_grad},
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f'Optimizer of type {self.config["training"]["optimizer"]["name"]} not supported. Currently supported optimizer include (Adam)'
            )

        print("Initializing loss function")
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.vocab_dict["<pad>"], reduction="mean"
        )

        print("Initializing tensorboard summary writer...")
        self.writer = SummaryWriter(log_dir=self.config["training"]["output_dir"])
        self.validation_steps = 0
        self.training_steps = 0
        self.repetition_penalty = self.config["visualisation"]["repetition_penalty"]
        self.temperature = self.config["visualisation"]["temperature"]

    def collate_fn(self, inputs):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in inputs])
        return {
            "input_ids": input_ids,
        }

    def save_checkpoint(self, checkpoint_name):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_fn": self.loss_fn,
            "training_steps": self.training_steps,
            "validation_steps": self.validation_steps,
            "config": self.config,
        }
        os.makedirs(
            os.path.join(self.config["training"]["output_dir"], "checkpoints"),
            exist_ok=True,
        )
        torch.save(
            checkpoint,
            os.path.join(
                self.config["training"]["output_dir"],
                "checkpoints",
                checkpoint_name + ".pth",
            ),
        )

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_validation_loss = 0.0
        validation_progress_bar = tqdm(
            self.validation_dataloader,
            desc=f"Epoch {epoch} / {self.config['training']['epochs']}",
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
        return validation_loss

    @torch.no_grad()
    def generate_one_sample(self, sample):
        input_tokens = sample["tokens"][:5]
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
        log_probs = []
        while len(generated_tokens) <= len(labels):
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
            last_output = outputs[:, -1, :].squeeze(0)
            mapped_token = torch.multinomial(
                torch.softmax(last_output / self.temperature, dim=-1), num_samples=1
            )[0].item()
            generated_tokens.append(mapped_token)
            if mapped_token == self.vocab_dict["<eos>"]:
                break
        prediction = " ".join(
            input_tokens + [self.inverse_vocab_dict[str(x)] for x in generated_tokens]
        )
        return prediction

    @torch.no_grad()
    def generate(self, epoch, num_samples=10):
        self.model.eval()
        generated_sentences = []
        original_continuations = []
        table = wandb.Table(columns=["input_text", "generated_text"])
        if num_samples != -1:
            ds = self.original_dataset["validation"].select(range(num_samples))
        else:
            ds = self.original_dataset["validation"]
        for sample in tqdm(ds):
            original_continuations.append(" ".join(sample["tokens"]))
            prediction = self.generate_one_sample(sample)
            generated_sentences.append(prediction)

        for input_text, generated_text in zip(
            original_continuations, generated_sentences
        ):
            self.writer.add_text("generate/input_text", input_text, epoch)
            self.writer.add_text("generate/generated_text", generated_text, epoch)
            table.add_data(input_text, generated_text)
        wandb.log({"generate/table": table}, step=epoch)

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
        self.writer.add_scalar("generate/bleu1", bleu_results1["bleu"], epoch)
        self.writer.add_scalar("generate/bleu2", bleu_results2["bleu"], epoch)
        wandb.log(
            {
                "generate/bleu1": bleu_results1["bleu"],
                "generate/bleu2": bleu_results2["bleu"],
            },
            step=epoch,
        )

    def train(self):
        best_validation_loss = float("inf")
        best_validation_epoch = -1
        epoch_time = []
        print("Training started....")
        for epoch in range(1, self.config["training"]["epochs"] + 1):
            start_time = time.time()
            training_epoch_loss = 0.0
            training_progress_bar = tqdm(
                self.pretrain_dataloader,
                desc=f"Epoch {epoch} / {self.config['training']['epochs']}",
            )
            self.model.train()
            for data in training_progress_bar:
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
                train_loss = self.loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
                maximum_memory_usage_before_backward = torch.cuda.max_memory_allocated() / 1024**2
                train_loss.backward()
                maximum_memory_usage_after_backward = torch.cuda.max_memory_allocated() / 1024**2
                training_epoch_loss += train_loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if (self.training_steps + 1) % self.config["training"][
                    "logging_interval"
                ] == 0:
                    self.writer.add_scalar(
                        "train/loss_per_step", train_loss.item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.training_steps,
                    )
                    wandb.log(
                        {
                            "train/loss_per_step": train_loss.item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.training_steps,
                    )
                    self.writer.add_scalar("train/maximum_memory_usage_before_backward", maximum_memory_usage_before_backward, self.training_steps)
                    self.writer.add_scalar("train/maximum_memory_usage_after_backward", maximum_memory_usage_after_backward, self.training_steps)
                    wandb.log(
                        {
                            "train/maximum_memory_usage_before_backward": maximum_memory_usage_before_backward,
                            "train/maximum_memory_usage_after_backward": maximum_memory_usage_after_backward,
                        },
                        step=self.training_steps,
                    )
                self.training_steps += 1
                training_progress_bar.set_postfix(loss=train_loss.item())

            end_time = time.time()
            epoch_time.append(end_time - start_time)
            training_epoch_loss = training_epoch_loss / len(self.pretrain_dataloader)
            training_epoch_perplexity = torch.exp(torch.tensor(training_epoch_loss))
            self.writer.add_scalar("train/loss_epoch", training_epoch_loss, epoch)
            self.writer.add_scalar(
                "train/perplexity_epoch_teacher_forcing",
                training_epoch_perplexity,
                epoch,
            )
            validation_epoch_loss = self.validate(epoch)
            validation_epoch_perplexity = torch.exp(torch.tensor(validation_epoch_loss))
            self.writer.add_scalar("valid/loss_epoch", validation_epoch_loss, epoch)
            self.writer.add_scalar(
                "valid/perplexity_epoch_teacher_forcing",
                validation_epoch_perplexity,
                epoch,
            )
            wandb.log(
                {
                    "train/loss_epoch": training_epoch_loss,
                    "train/perplexity_epoch_teacher_forcing": training_epoch_perplexity,
                },
                step=epoch,
            )
            wandb.log(
                {
                    "valid/loss_epoch": validation_epoch_loss,
                    "valid/perplexity_epoch_teacher_forcing": validation_epoch_perplexity,
                },
                step=epoch,
            )
            self.generate(epoch)
            if validation_epoch_loss < best_validation_loss:
                best_validation_loss = validation_epoch_loss
                best_validation_epoch = epoch
                self.save_checkpoint(f"best_checkpoint_{epoch}")

            if epoch - best_validation_epoch > self.config["training"]["patience"]:
                print(f"Early stopping triggered at epoch {epoch}")
                self.writer.close()
                wandb.finish()
                break

        with open(
            os.path.join(self.config["training"]["output_dir"], "epoch_time.json"), "w"
        ) as f:
            json.dump({"avg_epoch_time": np.mean(epoch_time)}, f)

        print("Training completed....")
        self.writer.close()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    print("Creating trainer...")
    trainer = Trainer(args.config_path)
    trainer.train()
