import os
import torch
from torch.utils.data import Dataset
import torchaudio


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

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        path = self.audio_files[idx]
        signal, sample_rate = torchaudio.load(path)
        speaker_id = self.speaker_ids[idx]
        transformed_signal = self._apply_transforms(signal)
        return transformed_signal, sample_rate, speaker_id

    def _apply_transforms(self, signal):
        for transform in self.transforms:
            signal = transform(signal)
        return signal
