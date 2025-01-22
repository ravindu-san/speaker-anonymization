import os

import torch.cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import torch

torch.cuda.empty_cache()
from tqdm import tqdm

from collator import Collator
from dataset import LibriSpeachDataset
from unet import Unet
from diffusion import Diffusion

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml


def save_model(model, file_path="./models/model.pt"):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    if not os.path.exists(file_path):
        open(file_path, "w").close()

    print("saving model...")
    torch.save(model.state_dict(), file_path)


def plot_losses(train_losses, val_losses, file="loss.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(len(train_losses)), train_losses, label="Training Loss", color="blue"
    )

    plt.plot(
        np.arange(len(val_losses)), val_losses, label="Validation Loss", color="red"
    )

    plt.title("Training vs Validation Loss", fontsize=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.yscale("log")
    plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    plt.savefig(file)


@torch.no_grad()
def test(model, test_loader, diffusor, device, args):
    model.eval()
    total_loss = 0.0
    timesteps = args["timesteps"]
    for step, (batch_spectrogram, batch_audio) in enumerate(test_loader):
        batch_spectrogram = batch_spectrogram.to(device)
        batch_audio = batch_audio.to(device)
        b_size = len(batch_spectrogram)
        t = torch.randint(0, timesteps, (b_size,), device=device).long()
        total_loss += diffusor.p_losses(
            model, batch_spectrogram, t, cond=batch_audio
        ).item()

        if dry_run:
            break

    avg_val_loss = total_loss / len(test_loader)
    print(f"Average validation loss: {avg_val_loss} \n")
    return avg_val_loss


def train(model, optimizer, train_loader, diffusor, epoch, device, args):
    model.train()
    total_loss = 0.0
    timesteps = args["timesteps"]
    dry_run = args["dry_run"]
    log_interval = args["log_interval"]
    pbar = tqdm(train_loader)
    for step, (batch_spectrogram, batch_audio) in enumerate(pbar):
        optimizer.zero_grad()
        batch_spectrogram = batch_spectrogram.to(device)
        batch_audio = batch_audio.to(device)
        b_size = len(batch_spectrogram)
        t = torch.randint(0, timesteps, (b_size,), device=device).long()
        loss = diffusor.p_losses(model, batch_spectrogram, t, cond=batch_audio)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    step * len(batch_spectrogram),
                    len(train_loader.dataset),
                    100.0 * step / len(train_loader),
                    loss.item(),
                )
            )

        total_loss += loss.item()

        if dry_run:
            break
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average train loss of epoch {epoch}: {avg_train_loss}")

    return avg_train_loss


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = args["epochs"]
    batch_size = args["batch_size"]
    lr = args["lr"]
    timesteps = args["timesteps"]
    schedule = args["schedule"]
    channels = args["channels"]
    hop_length = args["hop_length"]
    spectrogram_time_dim = args["spectrogram_time_dim"]

    dataset_train = LibriSpeachDataset(
        root_dir="./LibriSpeech", subset="train-clean-100", transforms=[]
    )
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collator(
            num_frames=spectrogram_time_dim, hop_length=hop_length
        ).collate,
    )

    dataset_val = LibriSpeachDataset(
        root_dir="./LibriSpeech", subset="test-clean", transforms=[]
    )
    validation_loader = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collator(
            num_frames=spectrogram_time_dim, hop_length=hop_length
        ).collate,
    )

    model = Unet(dim=32, channels=channels, dim_mults=(1, 2, 4, 8))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    diffusor = Diffusion(timesteps, schedule=schedule, device=device)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train(
            model,
            optimizer,
            train_loader=train_loader,
            diffusor=diffusor,
            epoch=epoch,
            device=device,
            args=args,
        )
        train_losses.append(train_loss)
        val_loss = test(
            model,
            test_loader=validation_loader,
            diffusor=diffusor,
            device=device,
            args=args,
        )
        val_losses.append(val_loss)

    if save_model:
        file_path = f"./models/model-{epochs}-{timesteps}-{schedule}.pt"
        save_model(model=model, file_path=file_path)

    loss_graphs_file = f"./loss_graphs/{timesteps}-{epochs}-{schedule}-{lr}.png"
    os.makedirs(os.path.dirname(loss_graphs_file), exist_ok=True)
    plot_losses(train_losses=train_losses, val_losses=val_losses, file=loss_graphs_file)


if __name__ == "__main__":
    args = yaml.safe_load(open("./config.yml"))

    epochs = args["epochs"]
    dry_run = args["dry_run"]

    print(f"Total number of epochs: {epochs},  dry run: {dry_run} \n\n")

    run(args=args)
