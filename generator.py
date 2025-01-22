import torch
from diffusion import Diffusion
from unet import Unet
import yaml
import os
from dataset import LibriSpeachDataset
from torch.utils.data import DataLoader
from collator import Collator
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_samples(model, diffusor, cond, shape, shift=5.513):
    shifted_generated_samples = diffusor.sample(model, cond, shape=shape)
    generated_samples = []
    for shifted_inter_sample in shifted_generated_samples:
        inter_sample = shifted_inter_sample * 6 - shift
        generated_samples.append(inter_sample)
    return generated_samples


def vocoder(sample, wav_file):
    tmpdir_vocoder = "tmpdir"
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz", savedir=tmpdir_vocoder
    )
    mel_specs = sample.squeeze()
    waveforms = hifi_gan.decode_batch(mel_specs)
    # waveforms_dir = 'generated_samples/waveforms'
    # os.makedirs(waveforms_dir, exist_ok=True)
    torchaudio.save(wav_file, waveforms.squeeze(1), 16000)
    print(f"saved reconstructed waveform in {wav_file}")


def visualize_spectrogram(
    spectrogram, save_img=False, file_name=None, hop_length=128
):  # note: use the same hop length as in sfft -> correct time
    # if index < spectrograms.shape[0]:
    if torch.is_tensor(spectrogram):
        spectrogram = spectrogram.cpu().numpy()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        spectrogram.squeeze(), x_axis="time", y_axis="mel", ax=ax, hop_length=hop_length
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    if save_img:
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {directory}: {e}")
        if not os.path.exists(file_name):
            open(file_name, "w").close()
        plt.savefig(file_name)
        # print('spectrogram image saved successfully...')


device = "cuda" if torch.cuda.is_available() else "cpu"

args = yaml.safe_load(open("./config.yml"))
timesteps = args["timesteps"]
schedule = args["schedule"]
hop_length = args["hop_length"]
channels = args["channels"]
sample_rate = args["sr"]
n_fft = args["n_fft"]
n_mels = args["n_mels"]
spectrogram_time_dim = args["spectrogram_time_dim"]
epochs = args["epochs"]
# n_spectrogram = args['n_spectrogram']
# n_spectrogram = 8

diffusor = Diffusion(timesteps, schedule=schedule, device=device)

model_path = f"./models/model-{epochs}-{timesteps}-linear.pt"
model = Unet(dim=32, channels=channels, dim_mults=(1, 2, 4, 8))
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

dataset = LibriSpeachDataset(
    root_dir="./LibriSpeech", subset="dev-clean", transforms=[]
)
data_loader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=Collator(num_frames=spectrogram_time_dim, hop_length=hop_length).collate,
)

batch_spectro, batch_audio = next(iter(data_loader))
while len(batch_audio) < 1:
    batch_spectro, batch_audio = next(iter(data_loader))
n_spectrogram = len(batch_audio)

generated_sample = generate_samples(
    model=model,
    diffusor=diffusor,
    cond=batch_audio,
    shape=(n_spectrogram, channels, n_mels, spectrogram_time_dim),
    shift=5.513,
)

samples = generated_sample[-1]
waveforms_dir = f"generated_samples/waveforms/after_gen/{epochs}-{timesteps}"
# waveforms_dir = 'generated_samples/waveforms'
os.makedirs(waveforms_dir, exist_ok=True)

# generate anonymized audio samples
for i, sample in enumerate(samples):
    wav_file = f"{waveforms_dir}/waveform_reconstructed-{i}.wav"
    vocoder(sample=sample, wav_file=wav_file)
    spectro_file = f"{waveforms_dir}/mel_spectro_reconstructed-{i}.png"
    visualize_spectrogram(
        sample, save_img=True, file_name=spectro_file, hop_length=hop_length
    )

# original audio samples
for i, audio in enumerate(batch_audio):
    wav_file = f"{waveforms_dir}/original-{i}.wav"
    torchaudio.save(wav_file, audio, 16000)
    spectro_file = f"{waveforms_dir}/original-{i}.png"
    visualize_spectrogram(
        batch_spectro[i] * 6 - 5.513,
        save_img=True,
        file_name=spectro_file,
        hop_length=hop_length,
    )
