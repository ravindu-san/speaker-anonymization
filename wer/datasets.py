import os
from pathlib import Path
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

class LibriSpeechAnonymized(Dataset):
    def __init__(self, root, subset):
        self._root = root
        self._subset = subset
        self._sample_rate = 16000
        self._ext_audio = ".flac"
        self._ext_txt = ".trans.txt"
        self._path = os.path.join(root, subset)
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))


    def __getitem__(self, n:int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid = self._walker[n]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        # Get audio path and sample rate
        fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
        filepath = os.path.join(self._root, self._subset, speaker_id, chapter_id, f"{fileid_audio}{self._ext_audio}")
        # waveform, sr = torchaudio.load(filepath)


        # Load text
        file_text = f"{speaker_id}-{chapter_id}{self._ext_txt}"
        file_text = os.path.join(self._root.replace("LibriSpeechAnonymized", "LibriSpeech"), "test-clean", speaker_id, chapter_id, file_text)
        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError(f"Translation not found for {fileid_audio}")

        return (
            # waveform,
            filepath,
            self._sample_rate,
            # sr,
            transcript,
            int(speaker_id),
            int(chapter_id),
            int(utterance_id),
        )


    def __len__(self) -> int:
        return len(self._walker)


class LibriSpeechAnonymizedDiffVC(Dataset):
    def __init__(self, root, subset):
        self._root = root
        self._subset = subset
        self._sample_rate = 16000
        self._ext_audio = ".flac"
        self._ext_txt = ".trans.txt"
        self._path = os.path.join(root, subset)
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))


    def __getitem__(self, n:int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid = self._walker[n]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        # Get audio path and sample rate
        fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
        filepath = os.path.join(self._root, self._subset, speaker_id, chapter_id, f"{fileid_audio}{self._ext_audio}")
        # waveform, sr = torchaudio.load(filepath)


        # Load text
        file_text = f"{speaker_id}-{chapter_id}{self._ext_txt}"
        file_text = os.path.join(self._root.replace("LibriSpeechAnonymized-DiffVC-16KHz", "LibriSpeech"), "test-clean", speaker_id, chapter_id, file_text)
        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError(f"Translation not found for {fileid_audio}")

        return (
            # waveform,
            filepath,
            self._sample_rate,
            # sr,
            transcript,
            int(speaker_id),
            int(chapter_id),
            int(utterance_id),
        )


    def __len__(self) -> int:
        return len(self._walker)



class LibriSpeech(Dataset):
    def __init__(self, root, subset):
        self._root = root
        self._subset = subset
        self._sample_rate = 16000
        self._ext_audio = ".flac"
        self._ext_txt = ".trans.txt"
        self._path = os.path.join(root, subset)
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))


    def __getitem__(self, n:int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid = self._walker[n]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        # Get audio path and sample rate
        fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
        filepath = os.path.join(self._root, self._subset, speaker_id, chapter_id, f"{fileid_audio}{self._ext_audio}")
        # waveform, sr = torchaudio.load(filepath)


        # Load text
        file_text = f"{speaker_id}-{chapter_id}{self._ext_txt}"
        file_text = os.path.join(self._root, self._subset, speaker_id, chapter_id, file_text)
        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError(f"Translation not found for {fileid_audio}")

        return (
            # waveform,
            filepath,
            self._sample_rate,
            # sr,
            transcript,
            int(speaker_id),
            int(chapter_id),
            int(utterance_id),
        )


    def __len__(self) -> int:
        return len(self._walker)