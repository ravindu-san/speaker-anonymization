import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
import matplotlib.pyplot as plt

from dataset import LibriSpeachDataset
from collator import Collator

# dataset = LIBRISPEECH("./", url="train-clean-100", download=True)
# waveform, sample_rate, _, _, _, _ = dataset[0]
#
# print(dataset[0])
# print(waveform.shape)
# if waveform.shape[0] > 1:
#     waveform = torch.mean(waveform, dim=0, keepdim=True)
#
# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12,8))
# ax1.plot(waveform.t().numpy())
# ax1.set_title('Waveform')
# ax1.set_ylim([-1,1])
#
# specgram = torchaudio.transforms.Spectrogram()(waveform)
# spec = specgram.log2()[0, :, :].detach().numpy()
# img = ax2.imshow(spec, aspect='auto', origin='lower')
# ax2.set_title('Spectrogram')
# # fig.colorbar(img, ax=ax2, format='%+2.0f dB')
#
#
# # plt.tight_layout()
# plt.show()

# dataset = LibriSpeachDataset(root_dir="./LibriSpeech")
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=128,
                                                                 n_mels=128)
dataset = LibriSpeachDataset(root_dir="./LibriSpeech", transforms=[mel_spectrogram_transform])
train_data = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=Collator().collate)
batch = next(iter(train_data))

print("end")
