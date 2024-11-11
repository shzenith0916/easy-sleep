import os
import numpy as np
import librosa
import soundfile as sf
import target_spectrogram_freq


def find_over_threshold_indices(S_DB, freq_indices, threshold_db):
    """Find indices that exceed the threshold in the specified frequency range.

    Parameters:
    - S_DB (numpy.ndarray): Decibel-scaled Mel spectrogram.
    - freq_indices (numpy.ndarray): Array of indices representing the target frequency range.
    - threshold_db (float): Threshold decibel value above which to find indices.

    Returns:
    - numpy.ndarray: Indices where the spectrogram exceeds the threshold dB.
    """
    threshold_indices = np.mean(S_DB[freq_indices, :], axis=0) > threshold_db

    return np.where(threshold_indices)[0]


def split_continuous_indices(over_threshold_indices):
    """Split indices into continuous segments.

    Parameters:
    - over_threshold_indices (numpy.ndarray): Array of indices that exceed the threshold.

    Returns:
    - list of numpy.ndarray: List of arrays, each representing a continuous segment of indices.
    """
    split_points = np.where(np.diff(over_threshold_indices) > 1)[0] + 1

    return np.split(over_threshold_indices, split_points)


def extract_audio_segments(audio, sr, segments):
    """Extract audio segments based on provided indices.

    Parameters:
    - audio (numpy.ndarray): Audio data.
    - sr (int): Sampling rate of the audio.
    - segments (list of numpy.ndarray): List of arrays, each array is a segment of indices.

    Returns:
    - list of numpy.ndarray: Extracted audio segments.
    """
    audio_segments = []
    for segment in segments:
        if len(segment) > 1:  # Minimum length condition
            start_sample = librosa.frames_to_samples(segment[0])
            end_sample = librosa.frames_to_samples(
                segment[-1] + 1)  # Include last frame

            audio_segments.append(audio[start_sample:end_sample])
    return audio_segments


def save_and_validate_segments(audio_segments, sr):
    """Save audio segments to files and delete if shorter than 0.5 seconds.

    Parameters:
    - audio_segments (list of numpy.ndarray): Audio segments to save.
    - sr (int): Sampling rate of the audio.

    """
    for i, segment in enumerate(audio_segments):
        output_path = f'output_segment_{i}.wav'
        sf.write(output_path, segment, sr)
        duration = librosa.get_duration(filename=output_path)
        if duration < 0.5:
            os.remove(output_path)
            print(
                f"Deleted {output_path} due to insufficient duration ({duration} sec).")
        else:
            print(f"Saved {output_path} with duration {duration} sec.")


if __name__ == "__main__":
    # Set the path to the audio file and load it with no specific sample rate and in mono.
    file_path = os.path.join(os.getcwd(), 'output.wav')
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    # Generate a Mel spectrogram from the audio data and convert it to decibel scale
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Using Viaulize_spectrogram.py file, check the freq_min and freq_max you would like to select.
    freq_min = 200
    freq_max = 1100
    # Define frequency limits; the chosen range typically captures snoring sounds.
    freq_indices = target_spectrogram_freq.frequency_range(
        S_DB, freq_min, freq_max)

    # Calculate the average decibel level across the specified frequency range.
    target_band_db = target_spectrogram_freq.compute_target_band_db(
        S_DB, freq_indices)

    # Define the threshold for snoring detection as 10 dB above the average dB in the target band.
    increment = 10  # Decibel increment to define snoring threshold.
    threshold_db = np.mean(target_band_db) + increment

    # Find indices where the spectrogram exceeds the defined snoring threshold.
    over_threshold_indices = find_over_threshold_indices(
        S_DB, freq_indices, threshold_db)

    # Identify continuous segments of indices where snoring is detected, assuming snoring events are contiguous.
    segments = split_continuous_indices(over_threshold_indices)

    # Extract audio segments based on the identified snoring events for further analysis or processing.
    audio_segments = extract_audio_segments(audio, sr, segments)

    # Save the detected snoring segments to audio files and delete any files shorter than 0.5 seconds.
    save_and_validate_segments(audio_segments, sr)
