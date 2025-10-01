import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
from pathlib import Path
from scipy.ndimage import median_filter


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


def reduce_noise(data, sr):
    noise_sample = data[:sr]  # 처음 1초를 배경 소음으로 사용
    return nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)


# def process_stationary_only(audio_path, segment_mins=5):
#     """1단계: Stationary 노이즈 제거만 수행"""
#     # 오디오 로드
#     audio, sr = librosa.load(audio_path, sr=16000)
#     print(f"오디오 길이: {len(audio)/sr/3600:.1f} hours")

#     # 30분 단위로 기본 노이즈(변화하지 않는 노이즈) 제거
#     print(f"Stationary 노이즈 제거 중...")
#     stage1_audio = denoise_stationary(audio, sr, segment_mins=30)

#     return stage1_audio, sr


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
    quietest_start = min(chunk_energies, key=lambda x: x[1])[0]
    noise_sample = audio[quietest_start:quietest_start + chunk_size]

    print(
        f"가장 조용한 구간: {quietest_start/sr/60:.1f}분 ~ {(quietest_start+chunk_size)/sr/60:.1f}분")

    return noise_sample