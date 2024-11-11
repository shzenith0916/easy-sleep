import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
from pathlib import Path
from scipy.ndimage import median_filter


def load_audio_and_resample(file_path, sr):
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr


def remove_dc_offset(data):
    mean_value = np.mean(data)
    return data - mean_value


def rms_normalize_audio(data, target_rms=0.1):
    rms = np.sqrt(np.mean(data**2))
    # 1e-6을 추가하는 이유는, 0으로 나누는것을 방지하기 위함
    scaling_factor = target_rms / (rms + 1e-6)
    return data * scaling_factor


def create_mel_spectrogram(data, sr, n_fft, hop_length, n_mels):
    mel_spec = librosa.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return mel_spec


def mel_spec_to_db(mel_spec, ref=np.max):
    '''dB 스케일로 변환 (시각적으로 더 직관적)'''
    mel_spec_db = librosa.power_to_db(
        mel_spec, ref=np.max)
    return mel_spec_db


def median_filter_on_spectrogram(mel_spec_db, size=(1, 3)):
    # Apply median filter directly on the spectrogram.
    filtered_spec = median_filter(mel_spec_db, size=size)
    return filtered_spec


if __name__ == "__main__":

    input_dir = ''
    output_dir = ''
    os.makedirs(output_dir, exist_ok=True)

    # Mel-spectrogram 생성 변수
    n_fft = 2048  # Fourier Transform window size
    hop_length = 256  # window 간 이동 간격
    n_mels = 128  # mel-spectrogram의 주파수 대역 수

    for file_path in Path(input_dir).glob('*.wav'):
        try:
            audio, sample_rate = load_audio_and_resample(file_path, 16000)
            dc_offset_removed = remove_dc_offset(audio)
            rms_normalized_data = rms_normalize_audio(dc_offset_removed)

            mel_spectrogram = create_mel_spectrogram(
                rms_normalized_data, sample_rate, n_fft, hop_length, n_mels)
            db_scaled_mel = mel_spec_to_db(mel_spectrogram)

            # Size (1, 3) for temporal smoothing
            filtered_S_db = median_filter(db_scaled_mel, size=(1, 3))

            file_name = file_path.stem

            # Mel-spectrogram 시각화 및 이미지 저장
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(db_scaled_mel, sr=sample_rate, hop_length=hop_length,
                                     x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title("{} Mel-spectrogram".format(file_name))
            plt.tight_layout()
            plt.savefig('{}.png'.format(file_name))
            plt.close()
            print(f"Spectrogram Image Created for {file_name}")

        except Exception as e:
            print(f"Error Occured {file_path}: {e}")
