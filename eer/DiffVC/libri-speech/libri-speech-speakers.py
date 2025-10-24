import os
import numpy as np


def get_speaker_ids(path):
    speaker_ids = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and entry.isdigit():
            speaker_ids.append(int(entry))  # Convert to int for numeric sorting
    return sorted(speaker_ids)


# Save to numpy
def save_speaker_ids(speaker_ids, filename="speakers.npy"):
    np.save(filename, np.array(speaker_ids))


# Load from numpy
def load_speaker_ids(filename="speakers.npy"):
    return np.load(filename)


def unique_speakers(train_speakers, test_speakers):
    train_set = set(train_speakers)
    test_set = set(test_speakers)
    filtered_train_set = train_set - test_set
    return sorted(filtered_train_set)


if __name__ == "__main__":
    librispeech_train_100_path = "../../../../LibriSpeech/train-clean-100"
    speaker_ids = get_speaker_ids(librispeech_train_100_path)
    save_speaker_ids(speaker_ids, filename="librispeech_train_100_speakers.npy")

    librispeech_test_path = "../../../../LibriSpeech/test-clean"
    speaker_ids = get_speaker_ids(librispeech_test_path)
    save_speaker_ids(speaker_ids, filename="librispeech_test_speakers.npy")

    train_100_speakers = load_speaker_ids(filename="librispeech_train_100_speakers.npy")
    test_speakers = load_speaker_ids(filename="librispeech_test_speakers.npy")

    unique_train_speakers = unique_speakers(train_100_speakers, test_speakers)
    unique_train_speakers_file = "librispeech_unique_train_speakers.npy"
    save_speaker_ids(unique_train_speakers, filename=unique_train_speakers_file)
    print(f"saved unique train speakers to {unique_train_speakers_file}")

    print(f"Train-100 Speaker Load: {train_100_speakers}")
    print(f"Test Speaker Load: {test_speakers}")
    print(f"Unique Speaker Load: {load_speaker_ids(unique_train_speakers_file)}")
    # print(f"Unique Speaker Set Diff: {Set()}")
