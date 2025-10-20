from typing import List

from datasets import LibriSpeechAnonymized, LibriSpeech, LibriSpeechAnonymizedDiffVC
import whisper
from jiwer import wer, process_words, visualize_alignment, Compose, RemovePunctuation, ToLowerCase, RemoveWhiteSpace
from tqdm import tqdm
import numpy as np

from scipy.stats import wilcoxon


import warnings
warnings.filterwarnings('ignore')


# os.environ["PATH"] += os.pathsep + "/usr/bin/ffmpeg"


def wer_librispeech_anonymized(epochs=20, subset="test-clean", experiment="librispeech"):

    # dataset = LibriSpeechAnonymized(root="/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeechAnonymized", subset="test-clean")
    dataset = LibriSpeechAnonymized(root="../../LibriSpeechAnonymized", subset=subset)

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
           f"Speaker model: Whisper (Large) | Dataset: LibriSpeech Anonymized (Our Method) | Subset: {subset}" \
           f"\n\n\tAverage WER over {epochs} repetitions: {(mean_WER) * 100:.2f} ± {(std_WER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(f'./results/{experiment}', 'a') as f:
        f.write(mesg)



def wer_librispeech(epochs=20, subset="test-clean", experiment="librispeech"):
    # dataset = LibriSpeech(root="/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeech", subset="test-clean")
    dataset = LibriSpeech(root="../../LibriSpeech", subset=subset)

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

        error = wer(references, hypothesis)
        word_error_rates.append(error)

    final_wer = np.stack(word_error_rates)
    mean_WER = final_wer.mean()
    std_WER = final_wer.std()

    print(f"Mean WER Librispeech: {mean_WER}")
    print(f"STD WER Librispeech: {std_WER}")

    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: Whisper (Large) | Dataset: LibriSpeech | Subset: {subset}" \
           f"\n\n\tAverage WER over {epochs} repetitions: {(mean_WER) * 100:.2f} ± {(std_WER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(f'./results/{experiment}', 'a') as f:
        f.write(mesg)
    

def wer_librispeech_diffvc_anonymized(epochs=20, subset="test-clean", experiment="librispeech"):

    # dataset = LibriSpeechAnonymizedDiffVC(root="/home/hpc/iwi5/iwi5248h/projects/Speaker Anonymization/LibriSpeechAnonymized-DiffVC-16KHz", subset=subset)
    dataset = LibriSpeechAnonymizedDiffVC(root="../../LibriSpeechAnonymized-DiffVC-16KHz", subset=subset)

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

        error = wer(references, hypothesis)
        word_error_rates.append(error)

    final_wer = np.stack(word_error_rates)
    mean_WER = final_wer.mean()
    std_WER = final_wer.std()

    print(f"Mean WER Librispeech DiffVC Anonymized: {mean_WER}")
    print(f"STD WER Librispeech DiffVC  Anonymized: {std_WER}")

    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: Whisper (Large) | Dataset: LibriSpeech DiffVC Anonymized | Subset: {subset}" \
           f"\n\n\tAverage WER over {epochs} repetitions: {(mean_WER) * 100:.2f} ± {(std_WER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(f'./results/{experiment}', 'a') as f:
        f.write(mesg)


def asr_outputs():
    asr_model = whisper.load_model("large")
    transform = Compose([
        RemovePunctuation(),
        ToLowerCase(),
        RemoveWhiteSpace(replace_by_space=True),
    ])

    references = []
    librispeech_hypothesis = []
    librispeech_anony_hypothesis = []
    librispeech_anony_diffvc_hypothesis = []
    librispeech_anony_diffvc_libritts_hypothesis = []

    dataset_librispeech = LibriSpeechAnonymized(root="../../LibriSpeech", subset="test-clean")
    dataset_librispeech_anony = LibriSpeechAnonymized(root="../../LibriSpeechAnonymized", subset="test-clean")
    dataset_librispeech_anony_diffvc = LibriSpeechAnonymizedDiffVC(root="../../LibriSpeechAnonymized-DiffVC-16KHz", subset="test-clean")
    dataset_librispeech_anony_diffvc_libritts = LibriSpeechAnonymizedDiffVC(root="../../LibriSpeechAnonymized-DiffVC-16KHz", subset="test-clean-libritts")


    for i, sample in enumerate(tqdm(dataset_librispeech)):
        waveform_file, _, transcript, *_ = sample
        waveform_file_anony = dataset_librispeech_anony[i][0]
        waveform_file_anony_diffvc = dataset_librispeech_anony_diffvc[i][0]
        waveform_file_anony_diffvc_libritts = dataset_librispeech_anony_diffvc_libritts[i][0]

        predicted_librispeech = asr_model.transcribe(waveform_file)
        predicted_librispeech_anony = asr_model.transcribe(waveform_file_anony)
        predicted_librispeech_anony_diffvc = asr_model.transcribe(waveform_file_anony_diffvc)
        predicted_librispeech_anony_diffvc_libritts = asr_model.transcribe(waveform_file_anony_diffvc_libritts)

        references.append(transform(transcript))
        librispeech_hypothesis.append(transform(predicted_librispeech["text"]))
        librispeech_anony_hypothesis.append(transform(predicted_librispeech_anony["text"]))
        librispeech_anony_diffvc_hypothesis.append(transform(predicted_librispeech_anony_diffvc["text"]))
        librispeech_anony_diffvc_libritts_hypothesis.append(transform(predicted_librispeech_anony_diffvc_libritts["text"]))

    np.save("references.npy", references)
    np.save("librispeech_hypothesis.npy", librispeech_hypothesis)
    np.save("librispeech_anony_hypothesis.npy", librispeech_anony_hypothesis)
    np.save("librispeech_anony_diffvc_hypothesis.npy", librispeech_anony_diffvc_hypothesis)
    np.save("librispeech_anony_diffvc_libritts_hypothesis.npy", librispeech_anony_diffvc_libritts_hypothesis)
    # return librispeech_wers, librispeech_anony_wers, librispeech_anony_diffvc_wers

#
# def get_error_counts(references: List[str], hypotheses: List[str]) -> np.ndarray:
#     error_data = []
#     for ref, hyp in zip(references, hypotheses):
#         measurement = process_words(ref, hyp)
#         error_count = (measurement.substitutions +
#                        measurement.deletions +
#                        measurement.insertions)
#
#         word_count = (measurement.hits +
#                       measurement.substitutions +
#                       measurement.deletions)
#
#         error_data.append([error_count, word_count])
#
#     return np.array(error_data)
#
#
# def paired_bootstrap_test(data: np.ndarray, B: int, N: int) -> np.ndarray:
#     delta_wers = []
#
#     for _ in range(B):
#         sample_indices = np.random.choice(N, size=N, replace=True)
#
#         # Slicing: [0/1=System, sample_indices=Utterances, 0/1=Count_Type]
#         sample_orig_counts = data[0, sample_indices, :]
#         sample_anon_counts = data[1, sample_indices, :]
#
#         total_errors_orig = np.sum(sample_orig_counts[:, 0])
#         total_words_orig = np.sum(sample_orig_counts[:, 1])
#
#         total_errors_anon = np.sum(sample_anon_counts[:, 0])
#         total_words_anon = np.sum(sample_anon_counts[:, 1])
#
#         wer_orig_star = total_errors_orig / total_words_orig
#         wer_anon_star = total_errors_anon / total_words_anon
#
#         delta_wer_star = wer_orig_star - wer_anon_star
#         delta_wers.append(delta_wer_star)
#
#     return np.array(delta_wers)
#
#
# def wer_pvalues(references: List[str], hypothesis_org: List[str], hypothesis_anony: List[str], bootstrap_samples: int) -> np.ndarray:
#     N = len(references)  # Total number of utterances/segments
#     B = bootstrap_samples  # Number of bootstrap samples
#     alpha = 0.05  # Significance level for a 95% Confidence Interval
#
#     counts_orig = get_error_counts(references, hypothesis_org)
#     counts_anon = get_error_counts(references, hypothesis_anony)
#
#     # Combine them for paired resampling: Shape (2, N, 2) -> (System, Utterance, [Error_Count, Word_Count])
#     paired_data = np.stack([counts_orig, counts_anon], axis=0)
#
#     delta_wers_distribution = paired_bootstrap_test(paired_data, B, N)
#
#     observed_wer_orig = wer(references, hypothesis_org)
#     observed_wer_anon = wer(references, hypothesis_anony)
#     observed_delta_wer = observed_wer_orig - observed_wer_anon
#
#     # Calculate the 95% Confidence Interval (Percentile Method)
#     ci_lower = np.percentile(delta_wers_distribution, 100 * (alpha / 2))
#     ci_upper = np.percentile(delta_wers_distribution, 100 * (1 - alpha / 2))
#
#     # Calculate the two-sided p-value
#     # P-value is the proportion of bootstrap samples where the difference is more extreme than the observed difference
#     p_value_two_sided = np.sum(np.abs(delta_wers_distribution) >= np.abs(observed_delta_wer)) / B
#
#     print(f"\n--- ASR Paired Bootstrap Test Results (B={B}) ---")
#     print(f"Observed WER Original (H_orig): {observed_wer_orig:.4f}")
#     print(f"Observed WER Anonymized (H_anon): {observed_wer_anon:.4f}")
#     print(f"Observed Delta WER (H_orig - H_anon): {observed_delta_wer:.4f}\n")
#
#     print("--- Statistical Significance ---")
#     print(f"95% Confidence Interval for Delta WER: ({ci_lower:.4f}, {ci_upper:.4f})")
#     print(f"Two-Sided P-Value: {p_value_two_sided:.4f}")
#     print(f"Significance level (alpha): {alpha}")
#
#     if 0 < ci_lower or 0 > ci_upper:
#         print("\n The difference is **statistically significant** (CI does not include 0).")
#     else:
#         print("\n The difference is **not statistically significant** (CI includes 0).")
#


def pvalue_wilcoxon(references, hypotheses_orig, hypotheses_anon, experiment):

    def get_error_counts(references: List[str], hypotheses: List[str]) -> np.ndarray:
        errors_per_sentence = []
        
        for ref, hyp in zip(references, hypotheses):
            measurement = process_words(ref, hyp)
            
            error_count = (measurement.substitutions + 
                        measurement.deletions + 
                        measurement.insertions)
            
            errors_per_sentence.append(error_count)
            
        return np.array(errors_per_sentence)

    errors_orig = get_error_counts(references, hypotheses_orig)
    errors_anon = get_error_counts(references, hypotheses_anon)

    try:
        statistic, p_value = wilcoxon(
            x=errors_orig, 
            y=errors_anon, 
            alternative='two-sided',
            correction=True 
        )

        print(f"--- Wilcoxon Signed-Rank Test Results: {experiment} ---")
        print(f"Number of Paired Observations (Sentences): {len(errors_orig)}")
        print(f"Wilcoxon W-Statistic: {statistic}")
        print(f"Two-Sided P-Value: {p_value:.2e}")
        print(f"Two-Sided P-Value: {repr(p_value)}")
        
        if p_value < 0.05:
            print("\nThe difference in the median error rate per sentence is **statistically significant** (p < 0.05).")
            
            overall_wer_orig = wer(references, hypotheses_orig)
            overall_wer_anon = wer(references, hypotheses_anon)
            
            if overall_wer_orig < overall_wer_anon:
                print("Conclusion: H_orig (Original) is significantly better than H_anon (Anonymized).")
            else:
                print("Conclusion: H_anon (Anonymized) is significantly better than H_orig (Original).")

        else:
            print("\nThe difference in the median error rate per sentence is **not statistically significant** (p >= 0.05).")

    except ValueError as e:
        print(f"\nCould not run Wilcoxon Test: {e}")
        print("Note: If all per-sentence error counts were identical (H_orig = H_anon), the test cannot be performed.")


if __name__ == '__main__':
    wer_librispeech(epochs=10)
    wer_librispeech_anonymized(epochs=10)
    wer_librispeech_diffvc_anonymized(epochs=10, subset="test-clean-libritts")
    wer_librispeech_diffvc_anonymized(epochs=10, subset="test-clean")

    # wer_per_utterance()

    np.random.seed(42)

    asr_outputs()

    references = np.load("references.npy").tolist() #original
    librispeech_hypothesis = np.load("librispeech_hypothesis.npy").tolist()
    librispeech_anony_hypothesis = np.load("librispeech_anony_hypothesis.npy").tolist()
    librispeech_anony_diffvc_hypothesis = np.load("librispeech_anony_diffvc_hypothesis.npy").tolist()
    librispeech_anony_diffvc_libritts_hypothesis = np.load("librispeech_anony_diffvc_libritts_hypothesis.npy").tolist()

    # print(len(references), len(librispeech_hypothesis))

    # wer_pvalues(references=references, hypothesis_org=librispeech_hypothesis, hypothesis_anony=librispeech_anony_hypothesis, bootstrap_samples=10000)
    # wer_pvalues(references=references, hypothesis_org=librispeech_hypothesis, hypothesis_anony=librispeech_anony_diffvc_hypothesis, bootstrap_samples=10000)
    # wer_pvalues(references=references, hypothesis_org=librispeech_anony_diffvc_hypothesis, hypothesis_anony=librispeech_anony_hypothesis, bootstrap_samples=10000)

    pvalue_wilcoxon(references=references, hypotheses_orig=librispeech_hypothesis, hypotheses_anon=librispeech_anony_hypothesis, experiment="org_vs_anony")
    pvalue_wilcoxon(references=references, hypotheses_orig=librispeech_hypothesis, hypotheses_anon=librispeech_anony_diffvc_hypothesis, experiment="org_vs_anony_diffvc_vctk")
    pvalue_wilcoxon(references=references, hypotheses_orig=librispeech_hypothesis, hypotheses_anon=librispeech_anony_diffvc_libritts_hypothesis, experiment="org_vs_anony_diffvc_libritts")
    pvalue_wilcoxon(references=references, hypotheses_orig=librispeech_anony_diffvc_hypothesis, hypotheses_anon=librispeech_anony_hypothesis, experiment="anony_vs_anony_diffvc_vctk")

