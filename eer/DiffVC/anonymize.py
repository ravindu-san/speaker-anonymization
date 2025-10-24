import os
import random
import glob
import soundfile as sf

import argparse
import json
import os
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch

use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn

mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import params
from model import DiffVC

import sys

sys.path.append("hifi-gan/")
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append("speaker_encoder/")
from encoder import inference as spk_encoder
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")


def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[: (wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode="reflect")
    stft = librosa.core.stft(
        wav, n_fft=1024, hop_length=256, win_length=1024, window="hann", center=False
    )
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i : i + 2 * w + 1])
        y[i] = min(x[i + w + 1], med)
    return y


def mel_spectral_subtraction(
    mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5
):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i : i + silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(
        np.exp(2.0 * mel_synth[:, i_min : i_min + silence_window]), axis=-1
    )
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(
            estimated_noise_energy, smoothing_window
        )
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(
            signal_subtract_noise, spectral_floor * estimated_noise_energy
        )
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


# # loading voice conversion model
# vc_path = 'checkpts/vc/vc_libritts_wodyn.pt' # path to voice conversion model

# generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
#                    params.layers, params.kernel, params.dropout, params.window_size,
#                    params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
#                    params.beta_min, params.beta_max)
# if use_gpu:
#     generator = generator.cuda()
#     generator.load_state_dict(torch.load(vc_path))
# else:
#     generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
# generator.eval()

# print(f'Number of parameters: {generator.nparams}')

# # loading HiFi-GAN vocoder
# hfg_path = 'checkpts/vocoder/' # HiFi-GAN path

# with open(hfg_path + 'config.json') as f:
#     h = AttrDict(json.load(f))

# if use_gpu:
#     hifigan_universal = HiFiGAN(h).cuda()
#     hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
# else:
#     hifigan_universal = HiFiGAN(h)
#     hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

# _ = hifigan_universal.eval()
# hifigan_universal.remove_weight_norm()


# # loading speaker encoder
# enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
# if use_gpu:
#     spk_encoder.load_model(enc_model_fpath, device="cuda")
# else:
#     spk_encoder.load_model(enc_model_fpath, device="cpu")


# loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
# src_path = 'example/6415_111615_000012_000005.wav' # path to source utterance
# tgt_path = 'example/8534_216567_000015_000010.wav' # path to reference utterance
# src_path = '/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac' # path to source utterance
# tgt_path = '/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech/dev-clean/174/50561/174-50561-0006.flac' # path to reference utterance

# mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
# if use_gpu:
#     mel_source = mel_source.cuda()
# mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
# if use_gpu:
#     mel_source_lengths = mel_source_lengths.cuda()

# mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
# if use_gpu:
#     mel_target = mel_target.cuda()
# mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
# if use_gpu:
#     mel_target_lengths = mel_target_lengths.cuda()

# embed_target = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
# if use_gpu:
#     embed_target = embed_target.cuda()


# # performing voice conversion
# mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
#                                       n_timesteps=30, mode='ml')
# mel_synth_np = mel_.cpu().detach().squeeze().numpy()
# mel_source_np = mel_.cpu().detach().squeeze().numpy()
# mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
# if use_gpu:
#     mel = mel.cuda()

# with torch.no_grad():
#     audio = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)


def is_audio_long_enough(file_path, min_duration_sec=1.0):
    """
    Checks if the audio file is at least `min_duration_sec` seconds long.
    """
    try:
        with sf.SoundFile(file_path) as f:
            duration = len(f) / f.samplerate
            return duration >= min_duration_sec
    except RuntimeError:
        return False  # Skip unreadable files


def select_random_audio_per_speaker(
    librispeech_root, subset="train-clean-100", min_duration_sec=1.0
):
    """
    Randomly selects one audio file per speaker from a LibriSpeech subset,
    ensuring each file is at least `min_duration_sec` seconds long.

    Returns:
        dict: Mapping of speaker ID to selected audio file path.
    """
    subset_path = os.path.join(librispeech_root, subset)
    if not os.path.isdir(subset_path):
        raise FileNotFoundError(f"Subset directory not found: {subset_path}")

    # speaker_to_file = {}
    selected_files = []

    for speaker_id in os.listdir(subset_path):
        speaker_path = os.path.join(subset_path, speaker_id)
        # print(speaker_path)
        if not os.path.isdir(speaker_path):
            continue

        # Find all .flac files under this speaker
        all_audio_files = glob.glob(os.path.join(speaker_path, "*", "*.flac"))
        # Filter by duration
        valid_audio_files = [
            f for f in all_audio_files if is_audio_long_enough(f, min_duration_sec)
        ]

        if not valid_audio_files:
            continue  # Skip speakers with no valid audio

        selected_file = random.choice(valid_audio_files)
        selected_files.append(selected_file)
        # speaker_to_file[speaker_id] = selected_file

    return selected_files


if __name__ == "__main__":
    librispeech_dir = "../../../LibriSpeech"
    selected_tgt_files = select_random_audio_per_speaker(
        librispeech_dir, subset="train-clean-100"
    )
    # print(len(selected_files))
    # for file_path in selected_files:
    #     print(file_path)

    # loading voice conversion model
    # vc_path = 'checkpts/vc/vc_libritts_wodyn.pt' # path to voice conversion model
    vc_path = "checkpts/vc/vc_vctk_wodyn.pt"  # path to voice conversion model

    generator = DiffVC(
        params.n_mels,
        params.channels,
        params.filters,
        params.heads,
        params.layers,
        params.kernel,
        params.dropout,
        params.window_size,
        params.enc_dim,
        params.spk_dim,
        params.use_ref_t,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
    )
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_path))
    else:
        generator.load_state_dict(torch.load(vc_path, map_location="cpu"))
    generator.eval()

    print(f"Number of parameters: {generator.nparams}")

    # loading HiFi-GAN vocoder
    hfg_path = "checkpts/vocoder/"  # HiFi-GAN path

    with open(hfg_path + "config.json") as f:
        h = AttrDict(json.load(f))

    if use_gpu:
        hifigan_universal = HiFiGAN(h).cuda()
        hifigan_universal.load_state_dict(
            torch.load(hfg_path + "generator")["generator"]
        )
    else:
        hifigan_universal = HiFiGAN(h)
        hifigan_universal.load_state_dict(
            torch.load(hfg_path + "generator", map_location="cpu")["generator"]
        )

    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()

    # loading speaker encoder
    enc_model_fpath = Path("checkpts/spk_encoder/pretrained.pt")  # speaker encoder path
    if use_gpu:
        spk_encoder.load_model(enc_model_fpath, device="cuda")
    else:
        spk_encoder.load_model(enc_model_fpath, device="cpu")

    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    # src_path = 'example/6415_111615_000012_000005.wav' # path to source utterance
    # tgt_path = 'example/8534_216567_000015_000010.wav' # path to reference utterance
    # src_path = '/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac' # path to source utterance
    # tgt_path = '/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech/dev-clean/174/50561/174-50561-0006.flac' # path to reference utterance

    speech_dir = (
        "/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech/test-clean"
    )

    for speaker in os.listdir(speech_dir):
        speaker_dir = os.path.join(speech_dir, speaker)
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            for file in os.listdir(chapter_dir):
                if not file.endswith(".flac"):
                    continue

                src_path = os.path.join(chapter_dir, file)
                tgt_path = random.choice(selected_tgt_files)
                anonymized_speech_file = src_path.replace(
                    "LibriSpeech", "LibriSpeechAnonymized-DiffVC-16KHz"
                )

                if os.path.isfile(anonymized_speech_file):
                    continue

                mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
                if use_gpu:
                    mel_source = mel_source.cuda()
                mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
                if use_gpu:
                    mel_source_lengths = mel_source_lengths.cuda()

                mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
                if use_gpu:
                    mel_target = mel_target.cuda()
                mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
                if use_gpu:
                    mel_target_lengths = mel_target_lengths.cuda()

                embed_target = (
                    torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
                )
                if use_gpu:
                    embed_target = embed_target.cuda()

                # performing voice conversion
                mel_encoded, mel_ = generator.forward(
                    mel_source,
                    mel_source_lengths,
                    mel_target,
                    mel_target_lengths,
                    embed_target,
                    n_timesteps=30,
                    mode="ml",
                )
                mel_synth_np = mel_.cpu().detach().squeeze().numpy()
                mel_source_np = mel_.cpu().detach().squeeze().numpy()
                mel = (
                    torch.from_numpy(
                        mel_spectral_subtraction(
                            mel_synth_np, mel_source_np, smoothing_window=1
                        )
                    )
                    .float()
                    .unsqueeze(0)
                )
                if use_gpu:
                    mel = mel.cuda()

                with torch.no_grad():
                    audio = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)

                audio = librosa.resample(
                    y=audio.numpy(), orig_sr=22050, target_sr=16000
                )
                os.makedirs(os.path.dirname(anonymized_speech_file), exist_ok=True)
                sf.write(
                    anonymized_speech_file,
                    audio,
                    16000,
                    format="flac",
                    subtype="PCM_24",
                )
