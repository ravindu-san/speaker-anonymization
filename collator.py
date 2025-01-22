import random
import torch
import numpy as np


class Collator:
    def __init__(self, num_frames, hop_length):
        self.num_frames = num_frames
        self.hop_length = hop_length

    def collate(self, batch):
        collated_batch_spectrograms = []
        collated_batch_audio = []
        for idx, batch_element in enumerate(batch):
            if batch_element[0].shape[-1] < self.num_frames:
                continue

            mel_spectrogram = batch_element[0]
            start = random.randint(0, mel_spectrogram.shape[-1] - self.num_frames)
            end = start + self.num_frames
            mel_spectrogram = mel_spectrogram[..., start:end]
            collated_batch_spectrograms.append(mel_spectrogram)

            audio_start = start * self.hop_length
            audio_end = end * self.hop_length
            audio = batch_element[1]
            audio = torch.squeeze(audio, 0)
            audio = audio[audio_start:audio_end]
            audio = np.pad(
                audio, (0, (audio_end - audio_start) - len(audio)), mode="constant"
            )
            audio = torch.unsqueeze(torch.tensor(audio), 0)
            collated_batch_audio.append(audio)

        return torch.stack(collated_batch_spectrograms), torch.stack(
            collated_batch_audio
        )
