import os
import numpy as np
import librosa


def load_audio_and_create_spectrogram(file_path):
    """Load an audio file and create its Mel spectrogram.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - audio (numpy.ndarray): Loaded audio data.
    - sr (int): Sampling rate of the audio.
    - S_DB (numpy.ndarray): Decibel-scaled Mel spectrogram.
    """
    # Load an audio 오디오 로드
    audio, sr = librosa.load(file_path, sr=None)
    # Generate a mel-spectrogram 멜 스펙트로그램 생성
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    # Convert to dB scale 데시벨 스케일로 변환
    S_DB = librosa.power_to_db(S, ref=np.max)

    return audio, sr, S, S_DB


def analyze_spectrogram(S_DB):
    """Analyze the spectrogram to find dB statistics."""
    max_db = np.max(S_DB)
    min_db = np.min(S_DB)
    mean_db = np.mean(S_DB)

    return max_db, min_db, mean_db


def frequency_range(S, freq_min, freq_max):
    """Calculate frequency indices within a specified range.

    Parameters:
    - S (numpy.ndarray): The Mel spectrogram array.
    - freq_min (float): The minimum frequency of the target range in Hz.
    - freq_max (float): The maximum frequency of the target range in Hz.

    Returns:
    - numpy.ndarray: An array of indices representing the specified frequency range within the Mel spectrogram.
    """
    # 주파수 범위 인덱스 계산
    freq_indices = np.where(
        (librosa.mel_frequencies(n_mels=S.shape[0]) >= freq_min) &
        (librosa.mel_frequencies(n_mels=S.shape[0]) <= freq_max)
    )[0]

    return freq_indices


def compute_target_band_db(S_DB, freq_indices):
    """Compute the average decibel (dB) level within a selected frequency band of a Mel spectrogram.

    Parameters:
    - S_DB (numpy.ndarray): Mel spectrogram in dB, with frequency bins as rows and time frames as columns.
    - freq_indices (numpy.ndarray or list): Indices of frequency bins within the target band.

    Returns:
    - float: Average dB level across the selected frequency band.
    """
    # 선택된 주파수 범위의 평균 데시벨 계산
    target_band_db = np.mean(S_DB[freq_indices, :], axis=0)
    return target_band_db


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "output.wav")

    audio, sr, S, S_DB = load_audio_and_create_spectrogram(file_path)

    max_db, min_db, mean_db = analyze_spectrogram(S_DB)

    print("Max dB:", max_db)
    print("Min dB:", min_db)
    print("Mean dB:", mean_db)

    # Get input from user or adjust the values as needed
    freq_min = float(
        input("Please type a target minimum frequency: "))  # e.g., 200
    freq_max = float(
        input("Please type a target maximum frequency: "))  # e.g., 1100

    freq_indices = frequency_range(S_DB, freq_min, freq_max)

    target_band_db = compute_target_band_db(S_DB, freq_indices)
    print("Average dB in target frequency band:", np.mean(target_band_db))
