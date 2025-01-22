import os
import torch
from torch.utils.data import Dataset
import torchaudio
import yaml


class LibriSpeachDataset(Dataset):
    def __init__(self, root_dir, subset="train-clean-100", transforms=None):
        self.root_dir = os.path.join(root_dir, subset)
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.audio_files = []
        self.speaker_ids = []

        for speaker_id in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker_id)
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                for file in os.listdir(chapter_dir):
                    if file.endswith(".flac"):
                        self.audio_files.append(os.path.join(chapter_dir, file))
                        self.speaker_ids.append(speaker_id)

        cfg = yaml.safe_load(open("./config.yml"))
        self.sr = cfg["sr"]
        self.n_mels = cfg["n_mels"]
        self.n_fft = cfg["n_fft"]
        self.hop_length = cfg["hop_length"]
        self.win_length = cfg["win_length"]
        self.power = cfg["power"]
        self.f_min = cfg["f_min"]
        self.f_max = cfg["f_max"]
        self.norm = cfg["norm"]
        self.mel_scale = cfg["mel_scale"]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        path = self.audio_files[idx]
        audio, sample_rate = torchaudio.load(path)
        speaker_id = self.speaker_ids[idx]

        audio = torch.clamp(audio, -1.0, 1.0)

        audio_to_mel = torchaudio.transforms.Spectrogram(
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            power=self.power,
            normalized=False,
        )

        mel_scale = torchaudio.transforms.MelScale(
            sample_rate=self.sr,
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=self.norm,
            mel_scale=self.mel_scale,
        )

        spec = audio_to_mel(audio)
        mel = mel_scale(spec)
        assert mel.dim() == 3
        assert mel.shape[1] == self.n_mels
        # mel = torch.log(torch.clamp(mel, min=1e-5) * 1)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = torch.clamp(mel, min=-11.513, max=0.487)
        mel = mel + 5.513
        mel = mel / 6
        return mel, audio, speaker_id

    def _apply_transforms(self, signal):
        for transform in self.transforms:
            signal = transform(signal)
        return signal
