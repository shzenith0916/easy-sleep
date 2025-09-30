import os
import glob
from pathlib import Path
import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def analyze_audio_stats(audio_input, sr):
    """
    오디오의 기본 통계 분석 (RMS, 최대 진폭, 다이나믹 레인지)

    rms: root mean square인데, 오디오 신호의 평균적인 음량을 나타내는 값
    max_amplitude: peak amplitude 최대 진폭 측정방법. 볼륨 레벨의 분석 또는 음성 활동 감지. 
    dynamic_range: 피크와 평균의 비율을 계산. 로그 변환 후 20*을  통해 dB 단위로 변환.(전력이 아닌 진폭 기준이므로 20 곱함)
    값이 클수록 조용한 부분과 큰 부분 차이가 크고, 값이 작을수록 비교적 일정함 볼륨(압축된 소리)
    duration_seconds: 오디오 신호의 길이를 초 단위로 나타내는 값
    duration_minutes: 오디오 신호의 길이를 분 단위로 나타내는 값
    """

    print(f"기본 오디오 통계 분석")

    if isinstance(audio_input, str):
        # 파일 경로이면,
        y, sr = librosa.load(audio_input, sr=sr)
    elif isinstance(audio_input, np.ndarray):
        # 이미 로드된 상황의 배열이면,
        y = audio_input
    else:
        raise TypeError(f"Invalid audio_input type: {type(audio_input)}")

    rms_level = np.sqrt(np.mean(y**2))
    max_amplitude = np.max(np.abs(y))
    dynamic_range = 20 * np.log10(max_amplitude / (rms_level + 1e-10))
    # (rms_level + 1e-10) 부분은 0으로 나누기를 방지하기 위한 작은값을 추가한 부분분

    stats = {
        'rms_level': rms_level,
        'max_amplitude': max_amplitude,
        'dynamic_range': dynamic_range,
        'duration_seconds': len(y) / sr,
        'duration_minutes': len(y) / sr / 60,
        'duration_hours': len(y) / sr / 3600
    }

    print(f"   - RMS Level: {rms_level:.4f}")
    print(f"   - Max Amplitude: {max_amplitude:.4f}")
    print(f"   - Dynamic Range: {dynamic_range:.1f} dB")
    print(f"   - Duration hours: {stats['duration_hours']:.1f}시간")

    return stats


def analyze_frequency_bands(audio_input, sr):
    """
    주파수 대역별 에너지 분석
    """

    if isinstance(audio_input, str):
        # 파일 경로이면,
        y, sr = librosa.load(audio_input, sr=sr)
    elif isinstance(audio_input, np.ndarray):
        # 이미 로드된 상황의 배열이면,
        y = audio_input
    else:
        raise TypeError(f"Invalid audio_input type: {type(audio_input)}")

    # STFT 계산
    stft = librosa.stft(y, n_fft=2048)
    magnitude = np.abs(stft)  # 2D array배열로, 주파수 빈도, 시간 프레임 값을 가짐.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    print(f"   - Magnitude shape: {magnitude.shape}")  # 디버깅용
    print(f"   - Frequency bins: {len(freqs)}")

    print(f"주파수 대역별 에너지 분석")

    # 주파수 대역 정의 딕셔너리
    freq_bands = {
        'Low-Freq(20-300Hz)': (20, 300),
        'Speech(300-2000Hz)': (300, 2000),
        'High-Freq(2000-8000Hz)': (2000, 8000),
        'Ultra-High(8000Hz+)': (8000, sr//2)
    }

    band_energies = {}

    for band_name, (low_freq, high_freq) in freq_bands.items():
        # 주파수를 인덱스로 변환. (주파수 범위를 찾아서 인덱스를 반환.)
        low_idx = np.searchsorted(freqs, low_freq)
        high_idx = np.searchsorted(freqs, high_freq)

        # 더 정확한 경계 체크
        max_freq_idx = len(freqs)  # 또는 magnitude.shape[0]

        # 빈 슬라이스 방지
        if low_idx >= high_idx or low_idx >= max_freq_idx:
            band_energy = 0.0
            band_energy_db = -np.inf  # 0의 dB는 -무한대
        else:
            high_idx = min(high_idx, max_freq_idx)
            band_magnitude = magnitude[low_idx:high_idx, :]
            band_energy = np.mean(band_magnitude)
            band_energy_db = 20 * np.log10(band_energy + 1e-10)

        band_energies[band_name] = band_energy
        print(
            f"   - {band_name}: {band_energy:.4f} (magnitude), {band_energy_db:.1f} dB")

    return band_energies
