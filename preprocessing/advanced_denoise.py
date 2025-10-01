import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
import matplotlib.pyplot as plt


def median_filter(mel_spec_db, size=(1, 3)):
    """스펙트로그램에 median filter 적용
    """
    filtered_spec = median_filter(mel_spec_db, size=size)
    return filtered_spec


class SnoringDenoiser:
    """
    코골이/호흡소리 전용 노이즈 제거기
    """

    def __init__(self, sr=16000):
        self.sr = sr
        # 코골이 주파수 범위 (Hz)
        self.snoring_min, self.snoring_max = 20, 500
        # 핵심 코골이 검출 대역 (Hz)
        # self.snoring_core_min, self.snoring_core_max = 30, 300
        # 말소리 제거 범위 (Hz)
        self.speech_min, self.speech_max = 2000, 4000
        # 기본 남성 여성 말 주파수
        # self.male_min, self.male_max = 85, 180
        # self.female_min, self.female_max = 165, 255

    def remove_background_noise(self, audio, noise_sample):
        """
        배경 소음 제거 (스펙트럴 서브트랙션)
        - noise_sample: 조용한 구간의 오디오 (노이즈 프로파일)
        """
        # 주파수 변환
        stft_audio = librosa.stft(audio, n_fft=2048, hop_length=512)
        stft_noise = librosa.stft(noise_sample, n_fft=2048, hop_length=512)

        # 노이즈 패턴 추정
        noise_pattern = np.mean(np.abs(stft_noise), axis=1, keepdims=True)

        # 노이즈 제거
        magnitude = np.abs(stft_audio)
        phase = np.angle(stft_audio)
        clean_magnitude = magnitude - 2.0 * noise_pattern
        clean_magnitude = np.maximum(clean_magnitude, 0.01 * magnitude)

        # 오디오 복원
        clean_stft = clean_magnitude * np.exp(1j * phase)
        return librosa.istft(clean_stft, hop_length=512)

    def remove_high_frequency(self, audio, cutoff_freq=2000):
        """
        고주파수 제거 (말소리, 전자기기 소음)
        - cutoff_freq: 차단 주파수 (기본값: 2000Hz)
        """
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist

        # 저역통과 필터 적용
        b, a = signal.butter(5, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)

    def keep_snoring_frequencies(self, audio):
        """
        코골이 주파수만 보존 (멜 스펙트로그램 기반)
        """
        # 멜 스펙트로그램 계산
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128, fmin=0, fmax=self.sr//2
        )

        # 코골이 주파수 범위만 선택
        mel_freqs = librosa.mel_frequencies(
            n_mels=128, fmin=0, fmax=self.sr//2)
        snoring_mask = (mel_freqs >= self.snoring_min) & (
            mel_freqs <= self.snoring_max)

        # 마스크 적용
        filtered_spec = mel_spec.copy()
        filtered_spec[~snoring_mask, :] = 0

        # 오디오로 복원
        return librosa.feature.inverse.mel_to_audio(filtered_spec, sr=self.sr)

    def extract_snoring_segments(self, audio, energy_threshold=75):
        """
        코골이 구간만 추출 (에너지 기반)
        - energy_threshold: 상위 몇 % 에너지를 가진 구간만 선택 (기본값: 75%)
        """
        # 에너지 계산
        hop_length = 512
        frame_length = 2048

        energy = np.array([
            np.sum(np.abs(audio[i:i+frame_length])**2)
            for i in range(0, len(audio), hop_length)
        ])

        # 고에너지 구간 찾기
        threshold = np.percentile(energy, energy_threshold)
        high_energy_frames = np.where(energy > threshold)[0]

        if len(high_energy_frames) > 0:
            start_sample = librosa.frames_to_samples(
                high_energy_frames[0], hop_length=hop_length)
            end_sample = librosa.frames_to_samples(
                high_energy_frames[-1], hop_length=hop_length)
            return audio[start_sample:end_sample + frame_length]
        else:
            return audio

    def clean_audio(self, audio, noise_sample=None):
        """
        전체 노이즈 제거 파이프라인 (간단한 3단계)
        1. 고주파 제거 (말소리)
        2. 배경 소음 제거  
        3. 코골이 주파수만 보존
        """
        print("노이즈 제거 시작...")

        # 1단계: 고주파 제거
        print("1단계: 고주파 제거 중...")
        step1 = self.remove_high_frequency(audio)

        # 2단계: 배경 소음 제거
        if noise_sample is not None:
            print("2️단계: 배경 소음 제거 중...")
            step2 = self.remove_background_noise(step1, noise_sample)
        else:
            step2 = step1

        # 3단계: 코골이 주파수만 보존
        print("3단계: 코골이 주파수 보존 중...")
        final = self.keep_snoring_frequencies(step2)

        print("노이즈 제거 완료!")
        return final

    def compare_results(self, original, cleaned):
        """
        노이즈 제거 전후 비교
        """
        # 에너지 변화
        original_energy = np.sum(original**2)
        cleaned_energy = np.sum(cleaned**2)
        energy_reduction = (1 - cleaned_energy/original_energy) * 100

        print(f"에너지 감소: {energy_reduction:.1f}%")
        print(f"신호 품질: {'개선됨' if energy_reduction > 0 else '유지됨'}")

        return energy_reduction


# 간단 사용 함수
def clean_snoring_audio(audio_file, sr=16000, noise_duration=20):
    """
    코골이 오디오 노이즈 제거 (간단한 사용법)

    Parameters:
    - audio_file: 오디오 파일 경로
    - sr: 샘플링 레이트
    - noise_duration: 노이즈 프로파일 길이 (초)
    """
    # 오디오 로드
    audio, sr = librosa.load(audio_file, sr=sr)

    # 노이즈 프로파일 (조용한 구간)
    noise_sample = audio[:noise_duration * sr]

    # 노이즈 제거
    denoiser = SnoringDenoiser(sr=sr)
    cleaned_audio = denoiser.clean_audio(audio, noise_sample)

    # 결과 비교
    denoiser.compare_results(audio, cleaned_audio)

    return cleaned_audio, sr


def show_waveform_comparison(original, cleaned, sr=16000, duration_minutes=2):
    """
    노이즈 제거 전후 파형 비교
    """
    max_samples = int(duration_minutes * 60 * sr)

    # 구간 제한
    if len(original) > max_samples:
        orig_seg = original[:max_samples]
        clean_seg = cleaned[:max_samples]
    else:
        orig_seg = original
        clean_seg = cleaned

    # 시간 축
    time_axis = np.linspace(0, len(orig_seg)/sr/60, len(orig_seg))

    # 그래프
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(time_axis, orig_seg, alpha=0.7, color='red')
    axes[0].set_title('Original Audio')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, clean_seg, alpha=0.7, color='blue')
    axes[1].set_title('After Noise Reduction')
    axes[1].set_xlabel('Time (Minutes)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":