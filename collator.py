import random


class Collator:
    def __init__(self):
        self.num_frames = 128

    # TODO only works for mel-spectrogram. need to generalize
    def collate(self, batch):
        for idx, batch_element in enumerate(batch):
            mel_spectrogram = batch_element[0]
            start = random.randint(0, mel_spectrogram.shape[-1] - self.num_frames)
            end = start + self.num_frames
            batch[idx] = (mel_spectrogram[..., start:end], *batch_element[1:])
        return batch
