import tqdm
import os
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as TT
import argparse


from glob import glob
from math import sqrt
from torch.nn import Linear, Conv1d, ConvTranspose2d, SiLU
from torch.utils.data.distributed import DistributedSampler
from fairseq import checkpoint_utils


AUDIO_FREQUNCY = 16000
AUDIO_TIME_LEN = 1
AUDIO_LEN = AUDIO_FREQUNCY * AUDIO_TIME_LEN
CROP_MEL_FRAMES = 62
# NOISE_SCHEDULE = np.linspace(1e-4, 0.05, 50)
NOISE_SCHEDULE = np.linspace(1e-4, 0.02, 200)
# NOISE_SCHEDULE = np.linspace(1e-4, 0.02, 1000)

LOSS_FN = nn.MSELoss()
FILTER_SIZE = 128  # residual channels
RES_LAYERS = 30


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, pickel_file, min_audio_len=AUDIO_LEN):
        super().__init__()
        self.utterances = []

        speaker_metadata = pickle.load(open(pickel_file, "rb"))

        for metadata in speaker_metadata:
            speaker = metadata[0]
            speaker_embedding = metadata[1]

            for utterance in metadata[2:]:
                file_name = utterance + ".flac"
                file_name = os.path.join(dataset_root, file_name)
                audio, _ = torchaudio.load(file_name)
                if audio.shape[-1] >= min_audio_len:
                    self.utterances.append((file_name, speaker, speaker_embedding))

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        audio_filename = utterance[0]
        audio, rate = torchaudio.load(audio_filename)
        audio = torch.clamp(audio[0], -1.0, 1.0)

        return audio, utterance[1], torch.from_numpy(utterance[2])


class ContentVecExtractor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0].eval()
        self.model = self.model.to(self.device)
        # self.model.eval()

    def extract_content_representations(self, waveforms):
        with torch.no_grad():
            waveforms = waveforms.to(self.device)
            content_rep = self.model.extract_features(waveforms)[0]
        return content_rep.transpose(-1, -2).cpu()


class Collator:
    def __init__(self, contentvec_extractor, audio_len=AUDIO_LEN):
        self.contentvec_extractor = contentvec_extractor
        self.audio_len = audio_len

    def collate(self, minibatch):
        audios = []
        speaker_emb = []
        for audio, _, speaker_embedding in minibatch:
            start = random.randint(0, audio.shape[-1] - self.audio_len)
            end = start + self.audio_len
            audios.append(audio[start:end])
            speaker_emb.append(speaker_embedding)

        audios_tensor = torch.stack(audios)

        contentvecs_tensor = self.contentvec_extractor.extract_content_representations(
            audios_tensor
        )

        speaker_embedding_tensor = torch.stack(speaker_emb)

        return {
            "audio": audios_tensor,
            "contentvec": contentvecs_tensor,
            "speaker_embedding": speaker_embedding_tensor,
        }


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """
    Positional Encoding
    detail could refer to:
    https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/2009.09761
    """

    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)

        x = self.projection1(x)
        x = SiLU()(x)
        x = self.projection2(x)
        x = SiLU()(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ContentVecUpsampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, contentvec, target_size):
        batch_size, feature_size, t = contentvec.shape

        x = contentvec.unsqueeze(1)

        x = F.interpolate(
            x,
            size=(feature_size, target_size),
            mode="nearest",  # bilinear or 'nearest' for faster but less smooth results
            # align_corners=False
        )

        x = x.squeeze(1)
        return x


class ResBlock(nn.Module):
    def __init__(self, res_channel, dilation, contentvec_feat_size, cond=True):
        super().__init__()
        self.dilated_conv = Conv1d(
            res_channel, 2 * res_channel, 3, padding=dilation, dilation=dilation
        )
        # self.diffstep_proj = Linear(512, res_channel)
        self.diffstep_proj = Linear(512, 2 * res_channel)
        self.cond_proj = Conv1d(
            contentvec_feat_size, 2 * res_channel, 1
        )  # 768 -> 256  or 512 -> 256
        self.output_proj = Conv1d(res_channel, 2 * res_channel, 1)
        self.cond = cond

        self.diff_step_plus_speker_trans = nn.Sequential(
            Linear(512, 256), SiLU(), Linear(256, res_channel), SiLU()
        )  # 256 == 2C here

    def forward(self, inp, diff_step, conditioner, speaker_emb):
        diff_step = self.diffstep_proj(diff_step)
        diff_step_plus_sp_emb = torch.concat((diff_step, speaker_emb), dim=1)
        diff_step_plus_sp_emb = self.diff_step_plus_speker_trans(
            diff_step_plus_sp_emb
        ).unsqueeze(-1)
        x = inp + diff_step_plus_sp_emb

        if self.cond:
            conditioner = self.cond_proj(conditioner)
            x = self.dilated_conv(x) + conditioner
        else:
            x = self.dilated_conv(x)

        gate, val = torch.chunk(x, 2, dim=1)  # gate function
        x = torch.sigmoid(gate) * torch.tanh(val)

        x = self.output_proj(x)
        residual, skip = torch.chunk(x, 2, dim=1)
        return (inp + residual) / np.sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, res_channels, n_layers, contentvec_feat_size, cond=True):
        super().__init__()
        self.cond = cond
        self.inp_proj = Conv1d(1, res_channels, 1)
        self.embedding = DiffusionEmbedding(len(NOISE_SCHEDULE))

        self.contentvec_upsampler = ContentVecUpsampler()

        self.contentvec_hidden_size = 512
        self.contentvec_feat_transform = nn.Sequential(
            nn.Conv1d(contentvec_feat_size, self.contentvec_hidden_size, 1), SiLU()
        )

        dilate_cycle = n_layers // 3
        self.layers = nn.ModuleList(
            [
                ResBlock(
                    res_channels,
                    2 ** (i % dilate_cycle),
                    self.contentvec_hidden_size,
                    self.cond,
                )
                for i in range(n_layers)
            ]
        )
        self.skip_proj = Conv1d(res_channels, res_channels, 1)
        self.output = Conv1d(res_channels, 1, 1)
        nn.init.zeros_(self.output.weight)

    def forward(self, audio, diffusion_step, contentvec, speaker_emb):
        x = audio.unsqueeze(1)  # (batch_size, 1, audio_sample)
        x = self.inp_proj(x)
        x = F.relu(x)
        diffusion_step = self.embedding(diffusion_step)

        contentvec = self.contentvec_feat_transform(contentvec)
        contentvec = self.contentvec_upsampler(contentvec, target_size=audio.shape[-1])

        skip = 0
        for layer in self.layers:
            x, skip_connection = layer(x, diffusion_step, contentvec, speaker_emb)
            skip += skip_connection

        x = skip / np.sqrt(len(self.layers))
        x = self.skip_proj(x)
        x = F.relu(x)
        x = self.output(x)
        return x


class DDPM(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.beta = NOISE_SCHEDULE
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha, 0)

    def forward(self, audio, t, noise):
        # xt = x0 * alpha_bar_sqrt + one_minus_alpha_bar * noise

        alpha_bar = torch.tensor(
            self.alpha_bar[t], device=self.device, dtype=torch.float32
        ).unsqueeze(1)
        alpha_bar_sqrt = alpha_bar**0.5
        one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        return alpha_bar_sqrt * audio + one_minus_alpha_bar * noise

    def reverse(self, x_t, pred_noise, t):
        alpha_t = np.take(self.alpha, t)
        alpha_t_bar = np.take(self.alpha_bar, t)

        mean = (1 / (alpha_t**0.5)) * (
            x_t - (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5 * pred_noise
        )

        if t == 0:
            return mean

        sigma = np.take(self.beta, t) ** 0.5
        z = torch.randn_like(x_t)
        return mean + sigma * z

    def generate(self, contentvec, speaker_emb, audio_len):
        if len(contentvec.shape) == 2:
            contentvec = contentvec.unsqueeze(0)
        if len(speaker_emb.shape) == 1:
            speaker_emb = speaker_emb.unsqueeze(0)

        contentvec = contentvec.to(self.device)
        speaker_emb = speaker_emb.to(self.device)

        x = torch.randn(contentvec.shape[0], audio_len, device=self.device)

        with torch.no_grad():
            for t in reversed(range(len(self.alpha))):
                t_tensor = torch.tensor(t, device=self.device).unsqueeze(0)
                pred_noise = self.model(x, t_tensor, contentvec, speaker_emb).squeeze(1)
                x = self.reverse(x, pred_noise, t)

        audio = torch.clamp(x, -1.0, 1.0)
        return audio


import os
import time
from tqdm import *


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        ckpt_dir,
        epochs,
        lr,
        save_n_epoch,
        diff_method,
        load_path=None,
    ):
        os.makedirs(ckpt_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.ckpt_dir = ckpt_dir

        self.best_epoch = 1
        self.start_epoch = 1
        self.epochs = epochs
        self.save_n_epoch = save_n_epoch

        self.diff_method = diff_method(self.model, self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = LOSS_FN

        # self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.model = nn.DataParallel(self.model)

        if load_path is not None:
            self.load_state_dict(load_path)
            print("sucessful load state dict !!!!!!")
            print(f"start from epoch {self.start_epoch}")

    def state_dict(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
        self.start_epoch = state_dict["epoch"]

    def train(self):
        for epoch in tqdm(
            range(self.start_epoch, self.epochs + 1), desc=f"Training progress"
        ):
            start = time.time()
            print(f"Start of epoch {epoch}")

            for i, audio_data in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                audio = audio_data["audio"].to(self.device)
                contentvec = audio_data["contentvec"].to(self.device)
                speaker_emb = audio_data["speaker_embedding"].to(self.device)
                noise = torch.randn_like(audio)
                t = np.random.randint(len(NOISE_SCHEDULE), size=audio.shape[0])

                noised_audio = self.diff_method(audio, t, noise)
                predict_noise = self.model(
                    noised_audio,
                    torch.tensor(t, device=self.device),
                    contentvec,
                    speaker_emb,
                ).squeeze()

                loss = self.loss_fn(noise, predict_noise)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1e9)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # if i > 500:
                #     break

            if self.best_epoch > loss:
                torch.save(self.state_dict(epoch), f"{self.ckpt_dir}/best_epoch.pt")
                print(
                    f"!!!!!!!!!!!!! saving best epoch {epoch} state dict !!!!!!!```````"
                )
                self.best_epoch = loss

            if epoch % self.save_n_epoch == 0:
                torch.save(
                    self.state_dict(epoch), f"{self.ckpt_dir}/weight_epoch{epoch}.pt"
                )
                print(f"sucessful saving epoch {epoch} state dict !!!!!!!")

            time_minutes = (time.time() - start) / 60
            print(f"epoch: {epoch}, loss: {loss.data} ~~~~~~")
            print(f"Time taken for epoch {epoch} is {time_minutes:.3f} min\n")

        print("finish training: ~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def generate(self, contentvec, speaker_emb, audio_len):
        return self.diff_method.generate(contentvec, speaker_emb, audio_len)


def parseArgs(parser):
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument(
        "--checkpoint_save_dir",
        default="./checkpoints/",
        type=str,
        help="directory where checkpoints are saved",
    )
    parser.add_argument(
        "--checkpoint_load_path",
        default=None,
        type=str,
        help="path to saved checkpoint which use as starting point for training",
    )
    parser.add_argument(
        "--checkpoint_every",
        default=5,
        type=int,
        help="frequency of saving checkpoints",
    )
    return parser


if __name__ == "__main__":
    print("Available GPUs: ", torch.cuda.device_count())
    print("GPU model:", torch.cuda.get_device_name(torch.cuda.current_device()))

    parser = argparse.ArgumentParser("train")
    parser = parseArgs(parser)
    args = parser.parse_args()
    print(args)

    batch_size = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    SAVE_PER_EPOCH = args.checkpoint_every
    SAVE_DIR = args.checkpoint_save_dir
    LOAD_PATH = args.checkpoint_load_path  # None for starting from epoch 1

    model = DiffWave(FILTER_SIZE, RES_LAYERS, 768)

    dataset = Dataset(
        dataset_root="../LibriSpeech/train-clean-100",
        pickel_file="./speaker_metadata.pkl",
    )

    contentvec_model_path = "checkpoint_best_legacy_500.pt"
    contentvec_extractor = ContentVecExtractor(model_path=contentvec_model_path)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Collator(contentvec_extractor=contentvec_extractor).collate,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    trainer = Trainer(
        model=model,
        dataloader=train_data,
        ckpt_dir=SAVE_DIR,
        epochs=EPOCHS,
        lr=LR,
        save_n_epoch=SAVE_PER_EPOCH,
        diff_method=DDPM,
        load_path=LOAD_PATH,
    )
    trainer.train()

    # # test_dataset = Dataset(root_dir="../LibriSpeech", subset="test-clean")
    # test_dataset = Dataset(dataset_root="../LibriSpeech/test-clean", pickel_file="./speaker_metadata_test.pkl")
    # test_data = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=2,
    #     collate_fn=Collator(contentvec_extractor=contentvec_extractor).collate,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # test_audio = next(iter(test_data))
    # result = trainer.generate(test_audio["contentvec"][0], test_audio['speaker_embedding'][5], audio_len=AUDIO_LEN)
    # result = result.data.cpu()
    #
    # os.makedirs("./diffwave_samples", exist_ok=True)
    #
    # wav_file = f"./samples/waveform_reconstructed.flac"
    # torchaudio.save(wav_file, result, 16000)
