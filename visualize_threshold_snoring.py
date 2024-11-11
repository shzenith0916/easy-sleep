import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import target_spectrogram_freq


def threshold_for_snoring(target_band_db, increment=10):
    """Calculate a threshold value based on the mean dB plus an increment.

    Parameters:
    - target_band_db (numpy.ndarray): Array of mean decibel levels across a specific frequency band.
    - increment (float): Decibel value to add to the mean dB to set the snoring threshold.

    Returns:
    - float: Calculated threshold in decibels.
    """

    threshold_db = np.mean(target_band_db) + increment  # 평균보다 10dB 높게 설정

    return threshold_db


def find_snoring_events(target_band_db, threshold_db):
    """Identify time indices where the dB level exceeds the threshold.

    Parameters:
    - target_band_db (numpy.ndarray): Array of decibel levels across a specific frequency band.
    - threshold_db (float): Threshold decibel level to identify significant snoring events.

    Returns:
    - numpy.ndarray: Array of times at which snoring events occur.
    """

    snoring_events = target_band_db > threshold_db
    times = librosa.frames_to_time(range(len(snoring_events)), sr=sr)

    return times[snoring_events]


def plot_snoring_events(audio, sr, snoring_events):
    """Plot the audio waveform and overlay snoring events.

    Parameters:
    - audio (numpy.ndarray): Audio time series data.
    - sr (int): Sampling rate of the audio.
    - snoring_times (numpy.ndarray): Times where snoring events were detected.
    """

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)

    plt.vlines(snoring_times, ymin=-1, ymax=1, color='r',
               linestyle='-', label='Snoring Detected')

    plt.title('Audio Waveform with Detected Snoring')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    file_path = os.path.join(os.getcwd(), 'output.wav')

    # Using functions from the imported module
    audio, sr, S, S_DB = target_spectrogram_freq.load_audio_and_create_spectrogram(
        file_path)
    max_db, min_db, mean_db = target_spectrogram_freq.analyze_spectrogram(S_DB)

    freq_indices = target_spectrogram_freq.frequency_range(
        S_DB, freq_min=200, freq_max=1100)
    target_band_db = target_spectrogram_freq.compute_target_band_db(
        S_DB, freq_indices)

    threshold_db = threshold_for_snoring(target_band_db)
    snoring_times = find_snoring_events(target_band_db, threshold_db)
    plot_snoring_events(audio, sr, snoring_times)
