import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
from pathlib import Path
from scipy.ndimage import median_filter
from scipy import signal


def denoise_stationary(audio, sr, segment_mins=30):
    """
    장시간 오디오를 30분 단위로 나누어서 stationary 노이즈 제거
    """
    segment_length = segment_mins * 60 * sr  # 30분 = 1800초
    processed_segments = []

    # start, end, stop
    for i in range(0, len(audio), segment_length):
        segment = audio[i: i + segment_length]

        # 각 30분 구간마다 stationary 노이즈 제거
        denoised_segment = nr.reduce_noise(
            y=segment,
            sr=sr,
            stationary=True,
            prop_decrease=0.7
        )

        processed_segments.append(denoised_segment)
        print(f"{segment_mins}분 세그먼트 처리: {i//segment_length +1}")

    return np.concatenate(processed_segments)

def denoise_non_stationary(audio, sr, window_mins=5):
    """비정상 노이즈 처리"""
    # size of audio chunk
    window_length = window_mins * 60 * sr
    processed_segments = []

    # start, end, stop interval(step)
    for i in range(0, len(audio), window_length):
        segment = audio[i: i + window_length]

        denoised_segment = nr.reduce_noise(
            y=segment,
            sr=sr,
            stationary=False,
            prop_decrease=0.8
        )
        processed_segments.append(denoised_segment)

    return np.concatenate(processed_segments)


def remove_dc_offset(data):
    mean_value = np.mean(data)
    return data - mean_value

def calculate_rms(audio):
    return np.sqrt(np.mean(audio**2))

def rms_normalize_audio(data, target_rms=0.1):
    rms = np.sqrt(np.mean(data**2))
    # 1e-6을 추가하는 이유는, 0으로 나누는것을 방지하기 위함
    scaling_factor = target_rms / (rms + 1e-6)
    return data * scaling_factor


def median_filter(self, audio, size=(61, 61)):
    """오디오에 median filter 적용
    
    parameters:
    - audio numpy.ndarray: 오디오 신호 (1D 배열)
    - size: median filter 크기  
    """

    filtered_audio = signal.medfilt(audio, size=size)
    return filtered_audio

def wiener_filter(self, audio):
    """오디오에 wiener filter 적용
    
    parameters:
    - audio numpy.ndarray: 오디오 신호 (1D 배열)
    """
    filtered_audio = signal.wiener(audio)
    return filtered_audio

def remove_high_frequency(self, audio, cutoff_freq=800):
    """
    고주파수 제거 (말소리 및 전자기기 소음 제거거)
    - cutoff_freq: 차단 주파수 (기본값: 2000Hz -> 800Hz)
    """
    nyquist = self.sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    # 저역통과 필터 적용 / Butterworth low-pass filter 설계
    b, a = signal.butter(5, Wn=normalized_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    print(f"말소리 및 고주파 제거 완료: {cutoff_freq}Hz 이상 제거")

    return filtered_audio

def noise_sample_reduction(data, sr):
    noise_sample = data[:sr]  # 처음 1초를 배경 소음으로 사용
    return nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)


def find_quiet_noise_profile(audio, sr, chunk_duration=5, percentile=10):
    """
    말소리 없는(에너지가 낮은) 조용한 구간을 소음 프로파일로 사용
    """
    # 5분 청크로 나누어 에너지 계산
    chunk_size = int(chunk_duration * sr)
    max_duration = min(len(audio), int(60 * 60 * sr)) # 최대 1시간

    chunk_data = []
    # 슬라이딩 윈도우로 더 많은 샘플 수집 (50% overlap)
    hop_size = chunk_size // 2

    for i in range(0, max_duration - chunk_size, hop_size):  # 첫 1시간만 확인
        chunk = audio[i:i+chunk_size]
        
        rms_energy = np.sqrt(np.mean(chunk**2))
       # zero-crossing rate (말소리 판단 기준)
        zcr = np.mean(librosa.zero_crossings(chunk))


    # 가장 조용한 청크 찾기
    quietest_start = min(chunk_data, key=lambda x: x[1])[0]
    noise_sample = audio[quietest_start:quietest_start + chunk_size]

    print(
        f"가장 조용한 구간: {quietest_start/sr/60:.1f}분 ~ {(quietest_start+chunk_size)/sr/60:.1f}분")

    return noise_sample