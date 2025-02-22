import os

import torch
import torchaudio

from diffwave import DiffWave, Trainer, DDPM, Dataset, Collator

# AUDIO_LEN = 22050*5
# CROP_MEL_FRAMES = 62 * 3
# NOISE_SCHEDULE = np.linspace(1e-4, 0.05, 50)

# LR = 5e-5
# LOSS_FN = nn.MSELoss()
EPOCHS = 300
SAVE_PER_EPOCH = 5
FILTER_SIZE = 128
RES_LAYERS = 30

SAVE_DIR = "./diffwave-training/"
# LOAD_PATH = "./diffwave-training/best_epoch.pt"
LOAD_PATH = "./diffwave-training/weight_epoch25.pt"

model = DiffWave(FILTER_SIZE, RES_LAYERS, 80)

trainer = Trainer(model, None, SAVE_DIR, EPOCHS, SAVE_PER_EPOCH, DDPM, LOAD_PATH)

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

# wav_file = f"./diffwave_samples/waveform_reconstructed_best_epoch.flac"
wav_file = f"./diffwave_samples/waveform_reconstructed.flac"
torchaudio.save(wav_file, result, 16000)

org_wav_file = f"./diffwave_samples/original.flac"
torchaudio.save(org_wav_file, test_audio['audio'][0].unsqueeze(0), 16000)