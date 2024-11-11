import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def load_audio_and_create_spectrogram(file_path):
    """Load an audio file and create its Mel spectrogram.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - audio (numpy.ndarray): Loaded audio data.
    - sr (int): Sampling rate of the audio.
    - S_DB (numpy.ndarray): Decibel-scaled Mel spectrogram.
    """

    # sr=None으로 설정: 오디오가 원래 샘플링 레이트로 로드되어서, 오디오 원본 특성 유지 가능.
    audio, sr = librosa.load(file_path, sr=None)
    # 오디오의 멜 스펙트로그램을 계산. 소리를 시간-주파수 형식으로 나타내는 것.
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    # C진폭 스펙트로그램을 dB 스케일 스펙트로그램으로 변환
    S_DB = librosa.power_to_db(S, ref=np.max)

    return audio, sr, S, S_DB


def visualize_spectrogram(S_DB, sr):
    """
    librosa.display.specshow: y axis indicates mel scale, x axis indicates time.
    """
    # 스펙트로그램 시각화
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(data=S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()


if __name__ == "__main__":

    file_path = os.path.join(os.getcwd(), 'output.wav')
    audio, sr, S, S_DB = load_audio_and_create_spectrogram(file_path)
    visualize_spectrogram(S_DB, sr)
