import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from . import audio_convert

# 한글 폰트 설정 (전역)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def visualize_waveform(file_path):
    """ 파형 시각화 함수
        오디오 파일 로드 후, librosa 라이브러리를 사용하여, 오디오 파형과 스펙트로그램을 시각화.

        parameters:
        - file_path: 오디오 파일 경로
        - sr: sampling rate를 의미. Default는 None. None일 경우, 오디오가 원래 샘플링 레이트로 로드되어서, 오디오 원본 특성 유지 가능.
        - audio 변수는 오디오 데이터를 의미

        librosa.display.waveshow를 사용하여 오디오의 파형을 시각화하여, 시간에 따른 소리의 크기와 강도를 이해하는데 도움.
        y축에 amplitude 진폭과 x축에 시간을 사용.  

        returns:
        - matplotlib figure 객체"""

    audio, sample_rate = audio_convert.Audio_Converter.load_audio_and_resample(file_path)

    fig = plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    return fig


def visualize_spectrogram(file_path):
    """ 스펙트로그램 시각화 """

    audio, sr = audio_convert.Audio_Converter.load_audio_and_resample(file_path)

    S = librosa.feature.melspectrogram(y=audio, sr=sr)

    S_DB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

    return fig


def save_figure(fig, output_path):
    """ figure 객체를 이미지 파일로 저장

    parameters:
    - fig: matplotlib figure 객체
    - output_path: 저장할 파일 경로
    """
    fig.savefig(output_path, dpi=300, bbox_inches='tight')


# =====================  test code =====================
audio_file = r"C:\Users\USER\Documents\코골이\easy_sleep\data\raw\20230904_SnoringSample_1.wav"

# 파형 시각화 및 저장
audio_file_ext_removed = audio_file.removesuffix(".wav")
waveform_save_path = audio_file_ext_removed + "_waveform.png"
fig1 = visualize_waveform(audio_file)
save_figure(fig1, waveform_save_path)

# 스펙트로그램 시각화 및 저장
spectrogram_save_path = audio_file_ext_removed + "_spectrogram.png"
fig2 = visualize_spectrogram(audio_file)
save_figure(fig2, spectrogram_save_path)
