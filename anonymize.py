import os
import random
import torch
import torchaudio
import torch.nn as nn
import numpy as np
from diffusion_contentvec import (
    DiffWave,
    DDPM,
    ContentVecExtractor,
    Trainer,
    Dataset,
    Collator,
)
from pseudo_speaker_generator import VAE
import pickle
from voicefixer import VoiceFixer
import tempfile
import shutil


FILTER_SIZE = 128  # residual channels
RES_LAYERS = 30
CONTENTVEC_FEAT_SIZE = 768

x_dim = 256
hidden_dim = 384
latent_dim = 64


def clean_file(input_path, output_path):
    vf = VoiceFixer()

    # print(f"Processing: {input_path}")
    vf.restore(input=input_path, output=output_path, cuda=torch.cuda.is_available())
    # print(f"Cleaned file saved to: {output_path}")


# def resample_audio(input_path, output_path, target_sr=16000):
def resample_audio(input_path, target_sr=16000):
    waveform, sr = torchaudio.load(input_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    resampled = resampler(waveform)
    # torchaudio.save(output_path, resampled, target_sr)
    return resampled


class Anonymizer:
    def __init__(self, diff_model_path, contentvec_model_path, speaker_emb_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.diffusion_syntheziser = DiffWave(
            FILTER_SIZE, RES_LAYERS, CONTENTVEC_FEAT_SIZE
        ).to(self.device)
        state_dict = torch.load(diff_model_path, map_location=self.device)
        self.diffusion_syntheziser.load_state_dict(state_dict["model"])

        self.contentvec_extractor = ContentVecExtractor(
            model_path=contentvec_model_path
        )

        self.pseudo_sp_embedder = VAE(x_dim, hidden_dim, latent_dim, self.device).to(
            self.device
        )
        checkpoint = torch.load(speaker_emb_model_path, map_location=self.device)
        self.pseudo_sp_embedder.load_state_dict(checkpoint["model"])

        self.ddpm = DDPM(self.diffusion_syntheziser, device=self.device)

    # def diffusion_anonymize(self, audio_file, speaker_emb=None):
    def diffusion_anonymize(self, audio, speaker_emb=None):
        # audio, _ = torchaudio.load(audio_file)
        # audio = torch.clamp(audio[0], -1.0, 1.0).unsqueeze(0)

        contentvec = self.contentvec_extractor.extract_content_representations(audio)

        if speaker_emb is None:
            with torch.no_grad():
                noise = torch.randn(1, 64).to(self.device)
                speaker_emb = self.pseudo_sp_embedder.Decoder(noise)

        anonymized_audio = self.ddpm.generate(
            contentvec=contentvec, speaker_emb=speaker_emb, audio_len=audio.shape[-1]
        )
        return anonymized_audio.data.cpu()

    # def anonymize(self, input_file, output_file, speaker_emb=None, save_intermediate=False):
    def anonymize(self, input_file, speaker_emb=None, save_intermediate=False):
        audio_, sr = torchaudio.load(input_file)
        audio = torch.clamp(audio_[0], -1.0, 1.0).unsqueeze(0)

        diff_anony_audio = self.diffusion_anonymize(
            audio=audio, speaker_emb=speaker_emb
        )

        # diff_anony_out_file = f"{output_file.split('/', )[0]}_diff_anony_out.wav"

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as diff_anony_out_file:
            diff_anony_out_file_path = diff_anony_out_file.name
        torchaudio.save(diff_anony_out_file_path, diff_anony_audio, sr)

        # cleaned_out_file = f"{output_file.split('.wav')[0]}_cleaned_44khz.wav"
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as cleaned_out_file:
            cleaned_out_file_path = cleaned_out_file.name
        clean_file(diff_anony_out_file_path, cleaned_out_file_path)

        # resample_audio(cleaned_out_file_path, output_file, sr)
        resampled_audio = resample_audio(cleaned_out_file_path, sr)

        if save_intermediate:
            org_file = f"{output_file.split('.wav')[0]}_org.wav"
            torchaudio.save(org_file, audio_, sr)

            final_diff_anony_out_file = (
                f"{output_file.split('.wav')[0]}_diff_anony_out.wav"
            )
            final_cleaned_44khz_file = (
                f"{output_file.split('.wav')[0]}_cleaned_44khz.wav"
            )
            shutil.move(diff_anony_out_file_path, final_diff_anony_out_file)
            shutil.move(cleaned_out_file_path, final_cleaned_44khz_file)
        else:
            os.remove(diff_anony_out_file_path)
            os.remove(cleaned_out_file_path)

        return resampled_audio


if __name__ == "__main__":
    diff_synth_model_path = "./checkpoints/weight_epoch4715.pt"
    # diff_synth_model_path = "./checkpoints/best_epoch.pt"
    contentvec_model_path = "./checkpoint_best_legacy_500.pt"
    speaker_emb_model_path = "./checkpoints/pseudo_speaker_vae/vae_model.ckpt"

    anonymizer = Anonymizer(
        diff_model_path=diff_synth_model_path,
        contentvec_model_path=contentvec_model_path,
        speaker_emb_model_path=speaker_emb_model_path,
    )

    audio_file = "../LibriSpeech/test-clean/61/70968/61-70968-0006.flac"
    output_file = (
        f"./generated_samples/anony/{audio_file.split('/')[-1].split('.')[0]}.wav"
    )

    # anonymizer.anonymize(audio_file, output_file, save_intermediate=True)
    anonymized_audio = anonymizer.anonymize(audio_file, save_intermediate=True)

    torch.save(output_file, anonymized_audio, 16000)

    # anonymized_audio = anonymizer.anonymize(audio_file=audio_file)

    # wav_file = f"./generated_samples/{audio_file.split('/')[-1].split('.')[0]}.wav"
    # torchaudio.save(wav_file, anonymized_audio, 16000)

    # wav_file_cleaned = f"./generated_samples/cleaned/{audio_file.split('/')[-1].split('.')[0]}_44khz.wav"
    # clean_file(wav_file, wav_file_cleaned)

    # wav_file_cleaned = f"./generated_samples/cleaned/{audio_file.split('/')[-1].split('.')[0]}.wav"
    # resample_audio(wav_file_cleaned, wav_file_cleaned, 16000)

    # print("finish anonymization...")
