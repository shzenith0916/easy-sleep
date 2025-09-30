import os
import numpy as np
import librosa
from audio_convert import load_audio_and_create_spectrogram


def analyze_spectrogram(S_DB):
    """멜 스펙트로그램의 최소값, 최대값, 평균값 계산
    Analyze the spectrogram to find dB statistics.

    Paramters:
    - S_DB (numpy.ndarray): Decibel-scaled Mel spectrogram
    """

    max_db = np.max(S_DB)
    min_db = np.min(S_DB)
    mean_db = np.mean(S_DB)

    return max_db, min_db, mean_db


def melBand_frequency_range_indices(mel_spec, sr=16000, freq_min=200, freq_max=1100):
    """멜 밴드 코골이 주파수 범위에 해당하는 인덱스 계산

    Parameters:
    - mel_spec (numpy.ndarray): Mel spectrogram
    - sr (int): Sampling rate
    - freq_min (float): Minimum frequency of the target range in Hz.
    - freq_max (float): Maximum frequency of the target range in Hz.

    Returns:
    - indices (numpy.ndarray): Indices of the frequency range
    """

    n_mels = mel_spec.shape[0]
    if freq_min is None:
        freq_min = 0.0
    if freq_max is None:
        freq_max = sr / 2.0

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    indices = np.where((mel_freqs >= freq_min) & (mel_freqs <= freq_max))[0]

    return indices


def compute_target_band_db(S_DB, freq_indices):
    """특정 주파수 범위 인덱스들의 평균 데시벨 계산
    Compute the average decibel (dB) level within a selected frequency band of a Mel spectrogram.

    Parameters:
    - S_DB (numpy.ndarray): Mel spectrogram in dB, with frequency bins as rows and time frames as columns.
    - freq_indices (numpy.ndarray or list): Indices of frequency bins within the target band.

    Returns:
    - float: Average dB level across the selected frequency band.
    """
    # 선택된 주파수 범위의 평균 데시벨 계산
    target_band_db = np.mean(S_DB[freq_indices, :], axis=0)
    print(f"코골이 주파수 범위의 평균 데시벨: {np.mean(target_band_db)}")

    return target_band_db


def threshold_dB(target_band_db, increment=10):
    """ 계산된 평균 데시벨 에서 increment 10정도 추가
    Add 10dB to the calculated mean dB

    Parameters:
    - target_band_db (numpy.ndarray): Array of decibel levels across a specific freq band.
    - increment (float): Decibel value to add to the mean dB to set the snoring threshold.
    """

    threshold_db = np.mean(target_band_db) + increment
    print("Threshold dB:", threshold_db)

    return threshold_db


def detect_snoring_events(target_band_db, threshold_db, sr=16000):
    """ 데시벨 임곗값 초과한 코골이 이벤트를 찾고, 시간 인덱스 반환 """

    snoring_events = target_band_db > threshold_db
    times = librosa.frames_to_time(range(len(snoring_events)), sr=sr)

    return times[snoring_events]


if __name__ == "__main__":
    file_path = r"C:\Users\USER\Documents\코골이\easy_sleep\data\raw\output.wav"

    audio, sr, S, S_DB = load_audio_and_create_spectrogram(file_path)

    max_db, min_db, mean_db = analyze_spectrogram(S_DB)

    print("Max dB:", max_db)
    print("Min dB:", min_db)
    print("Mean dB:", mean_db)

    freq_indices = melBand_frequency_range_indices(S_DB)

    target_band_db = compute_target_band_db(S_DB, freq_indices)

    threshold_db = threshold_dB(target_band_db)

    snoring_times = detect_snoring_events(target_band_db, threshold_db)

    print("DetectedSnoring times:", snoring_times)
