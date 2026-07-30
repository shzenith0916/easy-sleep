import os
import librosa
import numpy as np

# 멜 계산은 clip_features 에 단일 소스로 둔다. import 컨텍스트(패키지/flat)가
# 섞여 있어 상대 import 를 먼저 시도하고, 실패하면 flat 으로 폴백한다.
try:
    from .clip_features import clip_to_mel
except ImportError:
    from clip_features import clip_to_mel


class Audio_Converter():
    def __init__(self, sr=None, n_fft=2048, hop_length=256, n_mels=128):
        """ 오디오 변환 클래스 초기화 """
        self.sr = sr
        self.n_fft = n_fft  # Fourier Transform window size
        self.hop_length = hop_length  # window 간 이동 간격
        self.n_mels = n_mels  # mel-spectrogram의 주파수 대역 수

    @staticmethod
    def load_audio_and_resample(file_path, sr=None):
        """ 오디오 파일 로드하고, 샘플링 레이트 변경
        매개변수:
        - file_path (str): 오디오 파일 경로
        - sr (int): 샘플링 레이트
        반환: 
        - audio (numpy.ndarray): 오디오 데이터
        - sr (int): 샘플링 레이트
        예시) 16kHz샘플링, 10초 오디오 = 160000개의 샘플로 구성된 배열
        """
        # 오디오 로드
        audio, sr = librosa.load(file_path, sr=sr)
        print(f"오디오 로드 완료: {file_path}")
        print(f"샘플링 레이트: {sr}")

        return audio, sr

    @staticmethod
    def convert_to_mel(audio, sr, n_fft=2048, hop_length=256, n_mels=128):
        """ 오디오를 멜 스펙트로그램으로 변환. 소리를 시간-주파수 형식으로 나타내는 것.

        parameters:
        - audio (numpy.ndarray): Audio data
        - sr (int): sampling rate, 샘플링 레이트
        - n_fft (int): Fourier Transform window size, 푸리에 변환 창 크기
        - hop_length (int): Step size between windows, 윈도우 간 이동간격
        - n_mels (int): Number of Mel bands, 주파수 대역수

        returns:
        - mel_spec (numpy.ndarray): Mel spectrogram

        주의: 멜 계산은 clip_features.clip_to_mel 로 단일화돼 있다. (과거에는
        여기서 n_fft/hop_length/n_mels 를 받고도 librosa 에 넘기지 않아 hop=512
        기본값으로 동작하는 버그가 있었음 → 이제 인자가 그대로 반영된다.)
        모델 입력용 dB 멜이 필요하면 clip_features.clip_to_mel_db 를 쓸 것.
        """

        return clip_to_mel(audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    @staticmethod
    def convert_to_dB_scale(mel_spec):
        """ 멜 스펙트로그램을 데시벨 스케일로 변환 

        parameters:
        - mel_spec (numpy.ndarray): Mel spectrogram

        returns:
        - dB_mel_spec (numpy.ndarray): Decibel-scaled Mel spectrogram
        """
        dB_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return dB_mel_spec

    @staticmethod
    def load_audio_and_create_spectrogram(file_path):
        """Load an audio file and create its Mel spectrogram.

        Parameters:
        - file_path (str): Path to the audio file.

        Returns:
        - audio (numpy.ndarray): Loaded audio data.
        - sr (int): Sampling rate of the audio.
        - dB_mel_spec (numpy.ndarray): Decibel-scaled Mel spectrogram.
        """
        # Load an audio 오디오 로드
        audio, sr = librosa.load(file_path, sr=None)
        # Generate a mel-spectrogram 멜 스펙트로그램 생성
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        # Convert to dB scale 데시벨 스케일로 변환
        dB_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return audio, sr, mel_spec, dB_mel_spec
