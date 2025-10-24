import torch
import torch.nn as nn

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim import Adam
from pseudo_speaker_generator import VAE

import numpy as np
import pickle
import os
from tqdm import tqdm

# import matplotlib.pyplot as plt


# ENV settings
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

# Model Hyperparameters
batch_size = 32

x_dim = 256
hidden_dim = 384
latent_dim = 64

lr = 1e-3

epochs = 150


class SpkrEmbDataset(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, pkl_path):
        """Initialize and preprocess the Utterances dataset."""
        self.pkl_path = pkl_path

        """Load data"""
        self.load_data()

    def load_data(self):
        # load train.pkl
        with open(self.pkl_path, "rb") as f:
            meta = pickle.load(f)
        self.dataset = [(sbmt[0], sbmt[1]) for sbmt in meta]  # (spkr_id, spkr_emb)
        self.num_spkr = len(self.dataset)
        print("Finished loading the dataset...")

    def __getitem__(self, index):
        spkr_id, spkr_emb = self.dataset[index]
        return spkr_id, spkr_emb

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_spkr


def get_loader(pkl_path, batch_size=16, num_workers=0, shuffle=True, drop_last=True):
    """Build and return a data loader."""

    dataset = SpkrEmbDataset(pkl_path)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )
    return data_loader


# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(Encoder, self).__init__()
#         self.FC_input = nn.Linear(input_dim, hidden_dim)
#         # self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)

#         self.FC_mean = nn.Linear(hidden_dim, latent_dim)
#         self.FC_var = nn.Linear(hidden_dim, latent_dim)

#         self.LeakyReLU = nn.LeakyReLU(0.2)

#         self.training = True

#     def forward(self, x):
#         h_ = self.LeakyReLU(self.FC_input(x))
#         # h_ = self.LeakyReLU(self.FC_input2(h_))

#         mean = self.FC_mean(h_)
#         # encoder produces mean and log of variance
#         # (i.e., parateters of simple tractable normal distribution "q"
#         log_var = self.FC_var(h_)
#         return mean, log_var

# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
#         # self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
#         self.FC_output = nn.Linear(hidden_dim, output_dim)

#         self.LeakyReLU = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         h = self.LeakyReLU(self.FC_hidden(x))
#         # h = self.LeakyReLU(self.FC_hidden2(h))

#         x_hat = torch.tanh(self.FC_output(h))
#         return x_hat


# class VAE(nn.Module):
#     def __init__(self, x_dim, hidden_dim, latent_dim, DEVICE):
#         super(VAE, self).__init__()
#         self.Encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
#         self.Decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
#         self.DEVICE = DEVICE

#     def reparameterization(self, mean, var):
#         epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon
#         z = mean + var*epsilon  # reparameterization trick
#         return z

#     def forward(self, x):
#         mean, log_var = self.Encoder(x)
#         # takes exponential function (log var -> var)
#         z = self.reparameterization(mean, torch.exp(0.5 * log_var))
#         x_hat = self.Decoder(z)

#         return x_hat, mean, log_var


model = VAE(x_dim, hidden_dim, latent_dim, DEVICE).to(DEVICE)


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
mse = nn.MSELoss(reduction="sum")
l1 = nn.L1Loss(reduction="sum")


def loss_function(x, x_hat, mean, log_var):
    # cosine similarity loss
    cos_distance_loss = 200 * (1 - cos(x, x_hat)).sum()
    l1_loss = l1(x, x_hat)
    reconstruction_loss = cos_distance_loss + l1_loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss, KLD


librispeech_dataloader = get_loader(
    pkl_path="./speaker_metadata.pkl", num_workers=2, batch_size=batch_size
)

dataloader = librispeech_dataloader

print("Start training VAE...")
model.train()

optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    overall_loss, overall_reconst, overall_KLD = 0, 0, 0
    for batch_idx, (spkrids, x) in enumerate(dataloader):
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        reconst_loss, KLD = loss_function(x, x_hat, mean, log_var)
        loss = reconst_loss + KLD
        overall_loss += loss.item()
        overall_reconst += reconst_loss.item()
        overall_KLD += KLD.item()

        loss.backward()
        optimizer.step()

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\tAverage Loss: ",
        overall_loss / (batch_idx * batch_size),
        "\tAverage reconst Loss: ",
        overall_reconst / (batch_idx * batch_size),
        "\tAverage KLD Loss: ",
        overall_KLD / (batch_idx * batch_size),
    )

print("Finish!!")


eval_dataloader = get_loader(
    pkl_path="./speaker_metadata_test.pkl",
    num_workers=2,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)


def eval(model, dataloader):
    model.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = 0.0
    mse = 0.0
    n_sample = 0.0
    with torch.no_grad():
        for batch_idx, (spkrids, x) in enumerate(dataloader):
            x = x.to(DEVICE)
            n_sample += x.shape[0]
            x_hat, _, _ = model(x)
            cos_sim += cos(x_hat, x).sum()
            mse += ((x - x_hat) ** 2).sum()
    return {"cos_sim": cos_sim / n_sample, "mse": mse / n_sample}


print("eval librispeech test clean...:")
print("model:", eval(model, eval_dataloader))


torch.save(
    {"model": model.state_dict()}, "./checkpoints/pseudo_speaker_vae/vae_model.ckpt"
)


### no fine tuning


model = VAE(x_dim, hidden_dim, latent_dim, DEVICE).to(DEVICE)
checkpoint = torch.load(
    "./checkpoints/pseudo_speaker_vae/vae_model.ckpt", map_location=DEVICE
)
model.load_state_dict(checkpoint["model"])


with torch.no_grad():
    noise = torch.randn(batch_size, 64).to(DEVICE)
    generated_emb = model.Decoder(noise)

cos_sim = cos(generated_emb, x[4:5, :])
print(cos_sim, cos_sim.max(), cos_sim.mean())
