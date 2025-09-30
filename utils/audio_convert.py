import os
import librosa
import numpy as np


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
        """

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)

        return mel_spec

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
