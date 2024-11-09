import torchaudio
import torch
import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np


def analyze_librispeech(root_dir):
    stats = defaultdict(list)
    # subsets = ['train-clean-100', 'train-clean-360', 'train-other-500',
    #            'dev-clean', 'dev-other', 'test-clean', 'test-other']
    subsets = ['train-clean-100']
    for subset in subsets:
        subset_path = os.path.join(root_dir, subset)
        if not os.path.exists(subset_path):
            print(f"Skipping {subset} - not found")
            continue
        print(f"\nAnalyzing {subset}...")

        # Walk through all files in the subset
        for speaker_id in tqdm(os.listdir(subset_path)):
            speaker_path = os.path.join(subset_path, speaker_id)
            if not os.path.isdir(speaker_path):
                continue

            for chapter_id in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_id)
                if not os.path.isdir(chapter_path):
                    continue

                for audio_file in os.listdir(chapter_path):
                    if not audio_file.endswith('.flac'):
                        continue

                    file_path = os.path.join(chapter_path, audio_file)

                    try:
                        info = torchaudio.info(file_path)

                        duration = info.num_frames / info.sample_rate

                        stats['subset'].append(subset)
                        stats['speaker_id'].append(speaker_id)
                        stats['chapter_id'].append(chapter_id)
                        stats['duration'].append(duration)
                        stats['sample_rate'].append(info.sample_rate)
                        stats['num_channels'].append(info.num_channels)

                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")

    df = pd.DataFrame(stats)

    summary = {
        'total_hours': df['duration'].sum() / 3600,
        'total_speakers': df['speaker_id'].nunique(),
        'total_chapters': df['chapter_id'].nunique(),
        'avg_duration': df['duration'].mean(),
        'min_duration': df['duration'].min(),
        'max_duration': df['duration'].max(),
        'std_duration': df['duration'].std(),
        'max_channels': df['num_channels'].max(),
        'total_files': len(df),
    }

    subset_stats = df.groupby('subset').agg({
        'duration': ['sum', 'mean', 'std', 'count'],
        'speaker_id': 'nunique'
    })

    subset_stats.columns = ['total_duration', 'avg_duration', 'std_duration', 'num_files', 'num_speakers']
    subset_stats['hours'] = subset_stats['total_duration'] / 3600

    return df, summary, subset_stats


def print_statistics(summary, subset_stats):
    """Print formatted statistics"""
    print("\nOverall Dataset Statistics:")
    print(f"Total Hours: {summary['total_hours']:.2f}")
    print(f"Total Speakers: {summary['total_speakers']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Average Duration: {summary['avg_duration']:.2f} seconds")
    print(f"Duration Range: {summary['min_duration']:.2f} - {summary['max_duration']:.2f} seconds")
    print(f"Max Channels: {summary['max_channels']}")

    print("\nPer-Subset Statistics:")
    print(subset_stats[['hours', 'num_speakers', 'num_files', 'avg_duration']])


if __name__ == "__main__":
    LIBRISPEECH_ROOT = "LibriSpeech"

    df, summary, subset_stats = analyze_librispeech(LIBRISPEECH_ROOT)
    print_statistics(summary, subset_stats)

    # df.to_csv("librispeech_file_stats.csv", index=False)
    # subset_stats.to_csv("librispeech_subset_stats.csv")