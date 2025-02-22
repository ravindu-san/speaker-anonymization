import tqdm
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as TT
import  matplotlib.pyplot as plt


from glob import glob
from math import sqrt
from torch.nn import Linear, Conv1d, ConvTranspose2d, SiLU
from torch.utils.data.distributed import DistributedSampler


# AUDIO_LEN = 22050*5
AUDIO_LEN = 16000*5
CROP_MEL_FRAMES = 62
# NOISE_SCHEDULE = np.linspace(1e-4, 0.05, 50)
NOISE_SCHEDULE = np.linspace(1e-4, 0.02, 200)
# NOISE_SCHEDULE = np.linspace(1e-4, 0.02, 1000)

# LR = 5e-5
LR = 2e-4
LOSS_FN = nn.MSELoss()
EPOCHS = 250
SAVE_PER_EPOCH = 5
FILTER_SIZE = 128  # residual channels
RES_LAYERS = 30

SAVE_DIR = "./diffwave-training/"
LOAD_PATH = "./diffwave-training/weight_epoch225.pt"
# LOAD_PATH = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subset, cond=True):
        super().__init__()
        self.cond = cond
        self.filenames = []
        # for wav_file in glob(f'{path}/*.flac'):
        #     self.filenames.append(wav_file)
        self.root_dir = os.path.join(root_dir, subset)
        for speaker_id in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker_id)
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                for file in os.listdir(chapter_dir):
                    if file.endswith(".flac"):
                        self.filenames.append(os.path.join(chapter_dir, file))
                        # self.speaker_ids.append(speaker_id)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        audio, rate = torchaudio.load(audio_filename)
        audio = torch.clamp(audio[0], -1.0, 1.0)

        mel_args = {
            'sample_rate': rate,
            'win_length': 256 * 4,
            'hop_length': 256,
            'n_fft': 1024,
            'f_min': 20.0,
            'f_max': rate / 2.0,
            'n_mels': 80,
            'power': 1.0,
            'normalized': True,
        }
        mel_spec_transform = TT.MelSpectrogram(**mel_args)

        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)

        if self.cond:
            return {
                'audio': audio,
                'spectrogram': spectrogram.T
            }

        return {
            'audio': audio,
            'spectrogram': None
        }


class Collator():
    def __init__(self, cond=True):
        self.cond = cond

    def collate(self, minibatch):
        samples_per_frame = 256

        for record in minibatch:
            if not self.cond:
                # Filter out records that aren't long enough.
                if len(record['audio']) < AUDIO_LEN:
                    del record['spectrogram']
                    del record['audio']
                    continue

                start = random.randint(0, record['audio'].shape[-1] - AUDIO_LEN)
                end = start + AUDIO_LEN
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
            else:
                # Filter out records that aren't long enough.
                if len(record['spectrogram']) < CROP_MEL_FRAMES:
                    del record['spectrogram']
                    del record['audio']
                    continue

                start = random.randint(0, record['spectrogram'].shape[0] - CROP_MEL_FRAMES)
                end = start + CROP_MEL_FRAMES
                record['spectrogram'] = record['spectrogram'][start:end].T

                start *= samples_per_frame
                end *= samples_per_frame
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])

        if not self.cond:
            return {
                'audio': torch.from_numpy(audio),
                'spectrogram': None
            }

        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': torch.from_numpy(spectrogram)
        }

# import gc
# path = "./LibriSpeech/"
#
# dataset = Dataset(path)
# train_data = torch.utils.data.DataLoader(
#       dataset,
#       batch_size=16,
#       collate_fn=Collator().collate,
#       shuffle= True,
#       pin_memory=True,
#       drop_last=True)
#
# del dataset
# gc.collect()
#
# test_audio = next(iter(train_data))
# plt.plot(test_audio['audio'][0])
# plt.show()


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    '''
    Positional Encoding
    detail could refer to:
    https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/2009.09761
    '''

    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
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


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResBlock(nn.Module):
    def __init__(self, res_channel, dilation, n_mels, cond=True):
        super().__init__()
        self.dilated_conv = Conv1d(res_channel, 2 * res_channel, 3, \
                                   padding=dilation, dilation=dilation)
        self.diffstep_proj = Linear(512, res_channel)
        self.cond_proj = Conv1d(n_mels, 2 * res_channel, 1)
        self.output_proj = Conv1d(res_channel, 2 * res_channel, 1)
        self.cond = cond

    def forward(self, inp, diff_step, conditioner):
        diff_step = self.diffstep_proj(diff_step).unsqueeze(-1)
        x = inp + diff_step

        if self.cond:
            conditioner = self.cond_proj(conditioner)
            x = self.dilated_conv(x) + conditioner
        gate, val = torch.chunk(x, 2, dim=1)  # gate function
        x = torch.sigmoid(gate) * torch.tanh(val)

        x = self.output_proj(x)
        residual, skip = torch.chunk(x, 2, dim=1)
        return (inp + residual) / np.sqrt(2.0), skip



class DiffWave(nn.Module):
    def __init__(self, res_channels, n_layers, n_mels, cond=True):
        super().__init__()
        self.cond = cond
        self.inp_proj = Conv1d(1, res_channels, 1)
        self.embedding = DiffusionEmbedding(len(NOISE_SCHEDULE))
        self.spectrogram_upsampler = SpectrogramUpsampler(n_mels)

        dilate_cycle = n_layers // 3
        self.layers = nn.ModuleList([
            ResBlock(res_channels, 2 ** (i % dilate_cycle), n_mels, self.cond)
            for i in range(n_layers)
        ])
        self.skip_proj = Conv1d(res_channels, res_channels, 1)
        self.output = Conv1d(res_channels, 1, 1)
        nn.init.zeros_(self.output.weight)

    def forward(self, audio, diffusion_step, spectrogram):
        x = audio.unsqueeze(1)  # (batch_size, 1, audio_sample)
        x = self.inp_proj(x)
        x = F.relu(x)
        diffusion_step = self.embedding(diffusion_step)

        spectrogram = self.spectrogram_upsampler(spectrogram)
        if not self.cond:
            spectrogram = None

        skip = 0
        for layer in self.layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
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

        alpha_bar = torch.tensor(self.alpha_bar[t], device=self.device, \
                                 dtype=torch.float32).unsqueeze(1)
        alpha_bar_sqrt = alpha_bar ** 0.5
        one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        return alpha_bar_sqrt * audio + one_minus_alpha_bar * noise

    def reverse(self, x_t, pred_noise, t):
        alpha_t = np.take(self.alpha, t)
        alpha_t_bar = np.take(self.alpha_bar, t)

        mean = (1 / (alpha_t ** 0.5)) * (
                x_t - (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5 * pred_noise
        )
        sigma = np.take(self.beta, t) ** 0.5
        z = torch.randn_like(x_t)
        return mean + sigma * z

    def generate(self, spectrogram):
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(self.device)
        x = torch.randn(spectrogram.shape[0], 256 * spectrogram.shape[-1], device=self.device)

        with torch.no_grad():
            for t in reversed(range(len(self.alpha))):
                t_tensor = torch.tensor(t, device=self.device).unsqueeze(0)
                pred_noise = self.model(x, t_tensor, spectrogram).squeeze(1)
                x = self.reverse(x, pred_noise, t)
        audio = torch.clamp(x, -1.0, 1.0)
        return audio


import os
import time
from tqdm import *


class Trainer():
    def __init__(self, model, dataloader, ckpt_dir, \
                 epochs, save_n_epoch, diff_method, load_path=None):
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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = LOSS_FN

        if load_path is not None:
            self.load_state_dict(load_path)
            print("sucessful load state dict !!!!!!")
            print(f"start from epoch {self.start_epoch}")

    def state_dict(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict()
        }

    def load_state_dict(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.start_epoch = state_dict['epoch']

    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc=f"Training progress"):
            start = time.time()
            print(f'Start of epoch {epoch}')

            for i, audio_data in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                audio = audio_data['audio'].to(self.device)
                spectrogram = audio_data['spectrogram'].to(self.device)
                noise = torch.randn_like(audio)
                t = np.random.randint(len(NOISE_SCHEDULE), size=audio.shape[0])

                noised_audio = self.diff_method(audio, t, noise)
                predict_noise = self.model(noised_audio, \
                                           torch.tensor(t, device=self.device), spectrogram).squeeze()

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
                print(f"!!!!!!!!!!!!! saving best epoch {epoch} state dict !!!!!!!```````")
                self.best_epoch = loss

            if epoch % self.save_n_epoch == 0:
                torch.save(self.state_dict(epoch), f"{self.ckpt_dir}/weight_epoch{epoch}.pt")
                print(f"sucessful saving epoch {epoch} state dict !!!!!!!")

            time_minutes = (time.time() - start) / 60
            print(f"epoch: {epoch}, loss: {loss.data} ~~~~~~")
            print(f'Time taken for epoch {epoch} is {time_minutes:.3f} min\n')

        print("finish training: ~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def generate(self, spectrogram):
        return self.diff_method.generate(spectrogram)


if __name__ == "__main__":
    model = DiffWave(FILTER_SIZE, RES_LAYERS, 80)

    # import gc
    # path = "./LibriSpeech/train-clean-100"

    dataset = Dataset(root_dir="./LibriSpeech", subset="train-clean-100")
    train_data = torch.utils.data.DataLoader(
          dataset,
          batch_size=16,
          collate_fn=Collator().collate,
          shuffle= True,
          pin_memory=True,
          drop_last=True)

    # del dataset
    # gc.collect()
    #
    # test_audio = next(iter(train_data))
    # plt.plot(test_audio['audio'][0])
    # plt.show()

    trainer = Trainer(model, train_data, SAVE_DIR, EPOCHS,\
                       SAVE_PER_EPOCH, DDPM, LOAD_PATH)
    trainer.train()



    test_dataset = Dataset(root_dir="./LibriSpeech", subset="test-clean")
    test_data = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=16,
          collate_fn=Collator().collate,
          shuffle= True,
          pin_memory=True,
          drop_last=True)

    test_audio = next(iter(test_data))
    result = trainer.generate(test_audio['spectrogram'][0])
    result = result.data.cpu()

    os.makedirs('./diffwave_samples', exist_ok=True)

    wav_file = f"./diffwave_samples/waveform_reconstructed.flac"
    torchaudio.save(wav_file, result, 16000)

    # for i, sample in enumerate(result):
    #     wav_file = f"./diffwave_samples/waveform_reconstructed-{i}.wav"
    #     # vocoder(sample=sample, wav_file=wav_file)
    #     # spectro_file = f"{waveforms_dir}/mel_spectro_reconstructed-{i}.png"
    #     torchaudio.save(wav_file, sample, 16000)
    #     # visualize_spectrogram(
    #     #     sample, save_img=True, file_name=spectro_file, hop_length=hop_length
    #     # )

    # plt.plot(result)
    # plt.show()