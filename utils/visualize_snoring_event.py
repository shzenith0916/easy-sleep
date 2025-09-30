import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import target_spectrogram_freq


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
