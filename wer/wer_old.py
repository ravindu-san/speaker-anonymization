from datasets import LibriSpeechAnonymized, LibriSpeech
# from speechbrain.pretrained import EncoderDecoderASR
import whisper
from jiwer import wer, process_words, visualize_alignment, Compose, RemovePunctuation, ToLowerCase, RemoveWhiteSpace
from tqdm import tqdm
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# os.environ["PATH"] += os.pathsep + "/usr/bin/ffmpeg"


def wer_librispeech_anonymized(epochs=20, subset="test-clean", experiment="librispeech"):

    dataset = LibriSpeechAnonymized(root="/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeechAnonymized", subset=subset)

    asr_model = whisper.load_model("large")
    transform = Compose([
        RemovePunctuation(),
        ToLowerCase(),
        RemoveWhiteSpace(replace_by_space=True),
    ])

    word_error_rates = []

    for i in tqdm(range(epochs)):
        references = []
        hypothesis = []

        for sample in tqdm(dataset):
            waveform_file, _, transcript, *_ = sample
            
            # Get prediction
            predicted = asr_model.transcribe(waveform_file)

            references.append(transform(transcript))
            hypothesis.append(transform(predicted["text"]))

        # processed_words = process_words(references, hypothesis)
        error = wer(references, hypothesis)
        word_error_rates.append(error)
    # print(visualize_alignment(processed_words))
    final_wer = np.stack(word_error_rates)
    mean_WER = final_wer.mean()
    std_WER = final_wer.std()

    print(f"Mean WER Librispeech Anonymized (Our Method): {mean_WER}")
    print(f"STD WER Librispeech Anonymized (Our Method): {std_WER}")

    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: Whisper (Large) | Dataset: LibriSpeech Anonymized (Our Method | Subset: {subset})" \
           f"\n\n\tAverage WER over {epochs} repetitions: {(mean_WER) * 100:.2f} ± {(std_WER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(f'./results/{experiment}', 'a') as f:
        f.write(mesg)



def wer_librispeech(epochs=20, subset="test-clean", experiment="librispeech"):
    references = []
    hypothesis = []

    dataset = LibriSpeech(root="/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech", subset=subset)

    asr_model = whisper.load_model("large")
    transform = Compose([
        RemovePunctuation(),
        ToLowerCase(),
        RemoveWhiteSpace(replace_by_space=True),
    ])

    word_error_rates = []

    for i in tqdm(range(epochs)):
        references = []
        hypothesis = []

        for sample in tqdm(dataset):
            waveform_file, _, transcript, *_ = sample
            
            # Get prediction
            predicted = asr_model.transcribe(waveform_file)

            references.append(transform(transcript))
            hypothesis.append(transform(predicted["text"]))

        # processed_words = process_words(references, hypothesis)
        error = wer(references, hypothesis)
        word_error_rates.append(error)
    # print(visualize_alignment(processed_words))
    final_wer = np.stack(word_error_rates)
    mean_WER = final_wer.mean()
    std_WER = final_wer.std()

    print(f"Mean WER Librispeech (Our Method): {mean_WER}")
    print(f"STD WER Librispeech (Our Method): {std_WER}")

    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: Whisper (Large) | Dataset: LibriSpeech | Subset: {subset}" \
           f"\n\n\tAverage WER over {epochs} repetitions: {(mean_WER) * 100:.2f} ± {(std_WER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(f'./results/{experiment}', 'a') as f:
        f.write(mesg)
    

def wer_librispeech_diffvc_anonymized(epochs=20, subset="test-clean", experiment="librispeech"):

    dataset = LibriSpeechAnonymized(root="/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeechAnonymized-DiffVC-16KHz", subset="test-clean")

    asr_model = whisper.load_model("large")
    transform = Compose([
        RemovePunctuation(),
        ToLowerCase(),
        RemoveWhiteSpace(replace_by_space=True),
    ])

    word_error_rates = []

    for i in tqdm(range(epochs)):
        references = []
        hypothesis = []

        for sample in tqdm(dataset):
            waveform_file, _, transcript, *_ = sample
            
            # Get prediction
            predicted = asr_model.transcribe(waveform_file)

            references.append(transform(transcript))
            hypothesis.append(transform(predicted["text"]))

        # processed_words = process_words(references, hypothesis)
        error = wer(references, hypothesis)
        word_error_rates.append(error)
    # print(visualize_alignment(processed_words))
    final_wer = np.stack(word_error_rates)
    mean_WER = final_wer.mean()
    std_WER = final_wer.std()

    print(f"Mean WER Librispeech DiffVC Anonymized (Our Method): {mean_WER}")
    print(f"STD WER Librispeech DiffVC  Anonymized (Our Method): {std_WER}")

    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: Whisper (Large) | Dataset: LibriSpeech DiffVC Anonymized | Subset: {subset}" \
           f"\n\n\tAverage WER over {epochs} repetitions: {(mean_WER) * 100:.2f} ± {(std_WER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(f'./results/{experiment}', 'a') as f:
        f.write(mesg)


if __name__ == '__main__':
    wer_librispeech(epochs=10)
    wer_librispeech_anonymized(epochs=10)
    wer_librispeech_diffvc_anonymized(epochs=10)
