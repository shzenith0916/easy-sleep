"""긴 오디오용 디노이즈 함수 모음.

EDA 결론(notebooks/EDA.md)에 따라 **stationary 디노이즈**와 **저역통과**만 남겼다.
Wiener / raw-audio median filter 는 코골이 오디오에 부적합으로 결론나 제거했다.

주의: 모델 입력 전처리(B)에는 디노이즈를 넣지 않는다. utils/clip_features.py 의
clip_to_logmel 은 DC offset 제거만 한다. 여기 함수들은 **세그먼트 이전 단계의
긴 파일**에 선택적으로 쓰는 용도다.
"""
import numpy as np
import librosa
import noisereduce as nr
from scipy import signal


def denoise_stationary(audio, sr, segment_mins=30):
    """
    장시간 오디오를 30분 단위로 나누어서 stationary 노이즈 제거

    EDA3 비교에서 가장 무난했던 방법이다.
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


def remove_high_frequency(audio, sr, cutoff_freq=800):
    """고주파 제거 (말소리 및 전자기기 소음 제거)

    코골이 에너지는 저~중주파에 몰려 있고 2000Hz 이상은 거의 비어 있다는
    EDA2 관찰에 근거한다. 기본값을 2000Hz 에서 800Hz 로 좁혔다.

    parameters:
    - audio (numpy.ndarray): 오디오 신호 (1D 배열)
    - sr (int): 샘플레이트
    - cutoff_freq (int): 차단 주파수. 이보다 높은 성분을 제거한다.
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    # 저역통과 필터 적용 / Butterworth low-pass filter 설계
    b, a = signal.butter(5, Wn=normalized_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    print(f"말소리 및 고주파 제거 완료: {cutoff_freq}Hz 이상 제거")

    return filtered_audio


def noise_sample_reduction(data, sr):
    """맨 앞 1초를 배경 소음 프로파일로 써서 디노이즈.

    주의: 1초 이하의 짧은 클립에 쓰면 클립 자신이 자기 노이즈 프로파일이 되는
    degenerate 상태가 된다. 긴 파일에만 쓸 것. (EDA 의 feature_extract_test 참고)
    """
    noise_sample = data[:sr]  # 처음 1초를 배경 소음으로 사용
    return nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)


def find_quiet_noise_profile(audio, sr, chunk_duration=5, percentile=10):
    """말소리 없는(에너지가 낮은) 조용한 구간을 소음 프로파일로 반환한다.

    앞 1초를 무조건 쓰는 noise_sample_reduction 과 달리, 파일 안에서 실제로
    조용한 구간을 찾는다. 에너지(RMS)가 낮은 구간 중에서도 zero-crossing rate 가
    낮은 쪽을 고른다 — ZCR 이 높으면 말소리·마찰음이 섞여 있을 가능성이 크다.

    parameters:
    - audio (numpy.ndarray): 오디오 신호 (1D 배열)
    - sr (int): 샘플레이트
    - chunk_duration (float): 후보 구간 길이(초)
    - percentile (float): RMS 하위 몇 %를 후보로 삼을지. 이 안에서 ZCR 최소를 고른다.

    returns:
    - noise_sample (numpy.ndarray): 길이 chunk_duration*sr 의 노이즈 프로파일
    """
    chunk_size = int(chunk_duration * sr)
    max_duration = min(len(audio), int(60 * 60 * sr))  # 최대 1시간만 훑는다

    if chunk_size <= 0:
        raise ValueError("chunk_duration 이 0보다 커야 합니다.")
    if max_duration < chunk_size:
        raise ValueError(
            f"오디오가 너무 짧습니다: {len(audio)/sr:.1f}초 < chunk_duration {chunk_duration}초")

    # 슬라이딩 윈도우로 더 많은 샘플 수집 (50% overlap)
    hop_size = max(1, chunk_size // 2)

    chunk_data = []
    for i in range(0, max_duration - chunk_size + 1, hop_size):
        chunk = audio[i:i + chunk_size]

        rms_energy = np.sqrt(np.mean(chunk**2))
        # zero-crossing rate (말소리 판단 기준)
        zcr = np.mean(librosa.zero_crossings(chunk))

        chunk_data.append((i, rms_energy, zcr))

    # RMS 하위 percentile% 를 조용한 후보로 추리고, 그중 ZCR 이 가장 낮은 것을 고른다
    chunk_data.sort(key=lambda x: x[1])
    n_candidates = max(1, int(len(chunk_data) * percentile / 100))
    candidates = chunk_data[:n_candidates]

    quietest_start, quietest_rms, quietest_zcr = min(
        candidates, key=lambda x: x[2])
    noise_sample = audio[quietest_start:quietest_start + chunk_size]

    print(
        f"가장 조용한 구간: {quietest_start/sr/60:.1f}분 ~ "
        f"{(quietest_start+chunk_size)/sr/60:.1f}분 "
        f"(RMS {quietest_rms:.5f}, ZCR {quietest_zcr:.4f}, "
        f"후보 {n_candidates}/{len(chunk_data)}개 중)")

    return noise_sample
