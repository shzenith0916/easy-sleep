import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 한글 폰트 문제 해결
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# 1️⃣ 오디오 구간 추출 함수


def extract_audio_segment(audio_input, sr=22050, start_minutes=0, end_minutes=None, duration_minutes=None):
    """
    오디오에서 특정 구간만 추출

    Parameters:
    - audio_input: 파일 경로 또는 numpy array
    - sr: 샘플링 레이트
    - start_minutes: 시작 시간 (분)
    - end_minutes: 종료 시간 (분)
    - duration_minutes: 지속 시간 (분)

    Returns:
    - y_segment: 추출된 오디오 구간
    - sr: 샘플링 레이트
    - info: 구간 정보 딕셔너리
    """

    print(f"🔍 오디오 구간 추출")

    # 입력 타입 확인 및 처리
    if isinstance(audio_input, (str, Path)):
        print(f"파일에서 로드: {audio_input}")
        y_full, sr = librosa.load(audio_input, sr=sr)
        source_type = "파일"
    else:
        print(f"사전 처리된 오디오 데이터 사용")
        y_full = np.array(audio_input)
        source_type = "배열"

    total_duration_minutes = len(y_full) / sr / 60
    print(f"{source_type} 길이: {total_duration_minutes:.1f}분")

    # 구간 계산
    start_sample = int(start_minutes * 60 * sr)

    if end_minutes is not None:
        end_sample = int(end_minutes * 60 * sr)
        actual_duration = end_minutes - start_minutes
    elif duration_minutes is not None:
        end_sample = start_sample + int(duration_minutes * 60 * sr)
        actual_duration = duration_minutes
    else:
        end_sample = len(y_full)
        actual_duration = (len(y_full) - start_sample) / sr / 60

    # 범위 체크
    end_sample = min(end_sample, len(y_full))

    print(
        f"추출 구간: {start_minutes:.1f}분 ~ {start_minutes + actual_duration:.1f}분 (총 {actual_duration:.1f}분)")

    # 구간 추출
    y_segment = y_full[start_sample:end_sample]

    if len(y_segment) == 0:
        raise ValueError("지정된 구간이 비어있거나 범위를 벗어났습니다.")

    segment_info = {
        'start_minutes': start_minutes,
        'duration_minutes': actual_duration,
        'source_type': source_type,
        'total_duration_minutes': total_duration_minutes
    }

    print(f"✅ 추출 완료: {len(y_segment)/sr:.1f}초")
    return y_segment, sr, segment_info


# 2️⃣ 기본 오디오 통계 분석
def analyze_audio_stats(y_segment, sr):
    """
    오디오의 기본 통계 분석 (RMS, 최대 진폭, 다이나믹 레인지)
    """

    print(f"📊 기본 오디오 통계 분석")

    rms_level = np.sqrt(np.mean(y_segment**2))
    max_amplitude = np.max(np.abs(y_segment))
    dynamic_range = 20 * np.log10(max_amplitude / (rms_level + 1e-10))

    duration_seconds = len(y_segment) / sr
    duration_minutes = duration_seconds / 60

    stats = {
        'rms_level': rms_level,
        'max_amplitude': max_amplitude,
        'dynamic_range': dynamic_range,
        'duration_seconds': duration_seconds,
        'duration_minutes': duration_minutes
    }

    print(f"   - RMS Level: {rms_level:.4f}")
    print(f"   - Max Amplitude: {max_amplitude:.4f}")
    print(f"   - Dynamic Range: {dynamic_range:.1f} dB")
    print(f"   - Duration: {duration_seconds:.1f}초 ({duration_minutes:.2f}분)")

    return stats


# 3️⃣ 주파수 대역별 에너지 분석
def analyze_frequency_bands(y_segment, sr):
    """
    주파수 대역별 에너지 분석
    """

    print(f"🎵 주파수 대역별 에너지 분석")

    # STFT 계산
    stft = librosa.stft(y_segment, n_fft=2048)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # 주파수 대역 정의
    freq_bands = {
        'Low Freq (20-300Hz)': (20, 300),
        'Speech (300-2000Hz)': (300, 2000),
        'High Freq (2000-8000Hz)': (2000, 8000),
        'Ultra High (8000Hz+)': (8000, sr//2)
    }

    band_energies = {}

    for band_name, (low_freq, high_freq) in freq_bands.items():
        low_idx = np.searchsorted(freqs, low_freq)
        high_idx = np.searchsorted(freqs, high_freq)

        # 빈 슬라이스 방지
        if low_idx >= high_idx or low_idx >= len(magnitude):
            band_energy = 0.0
        else:
            high_idx = min(high_idx, len(magnitude))
            band_energy = np.mean(magnitude[low_idx:high_idx])

        band_energies[band_name] = band_energy
        print(f"   - {band_name}: {band_energy:.4f}")

    return band_energies


# 4️⃣ 노이즈 유형 추정
def estimate_noise_type(band_energies):
    """
    주파수 대역 에너지를 기반으로 노이즈 유형 추정
    """

    print(f"🎯 노이즈 유형 추정")

    total_energy = sum(band_energies.values())

    if total_energy > 0:
        speech_ratio = band_energies['Speech (300-2000Hz)'] / total_energy
        low_freq_ratio = band_energies['Low Freq (20-300Hz)'] / total_energy
        high_freq_ratio = band_energies['High Freq (2000-8000Hz)'] / \
            total_energy
    else:
        speech_ratio = low_freq_ratio = high_freq_ratio = 0.0

    # 노이즈 유형 분류
    if speech_ratio > 0.4:
        noise_type = "🗣️ Speech-containing segment"
    elif high_freq_ratio > 0.3:
        noise_type = "📺 TV/Electronic noise"
    elif low_freq_ratio > 0.4:
        noise_type = "🌬️ Environmental/Background noise"
    else:
        noise_type = "🔇 Quiet segment"

    ratios = {
        'speech_ratio': speech_ratio,
        'low_freq_ratio': low_freq_ratio,
        'high_freq_ratio': high_freq_ratio,
        'noise_type': noise_type
    }

    print(f"   - Type: {noise_type}")
    print(f"   - Speech Ratio: {speech_ratio:.1%}")
    print(f"   - Low Freq Ratio: {low_freq_ratio:.1%}")
    print(f"   - High Freq Ratio: {high_freq_ratio:.1%}")

    return ratios


# 5️⃣ 노이즈 제거 권장사항
def recommend_noise_reduction(stats, ratios):
    """
    통계와 주파수 분석을 바탕으로 노이즈 제거 권장사항 제시
    """

    print(f"💡 노이즈 제거 권장사항")

    rms_level = stats['rms_level']
    speech_ratio = ratios['speech_ratio']
    high_freq_ratio = ratios['high_freq_ratio']

    # 제거 강도 결정
    if rms_level > 0.1:
        print("   ⚠️ Very high noise level - Strong reduction needed")
        reduction_strength = "strong"
    elif rms_level > 0.05:
        print("   ⚡ Medium noise level - Moderate reduction")
        reduction_strength = "medium"
    else:
        print("   ✅ Low noise level - Light reduction sufficient")
        reduction_strength = "light"

    recommendations = {
        'reduction_strength': reduction_strength,
        'two_stage_needed': False,
        'lowpass_filter_needed': False
    }

    if speech_ratio > 0.3:
        print("   🎤 Speech detected - 2-stage processing recommended")
        recommendations['two_stage_needed'] = True

    if high_freq_ratio > 0.4:
        print("   📻 High frequency noise - 2000Hz low-pass filter recommended")
        recommendations['lowpass_filter_needed'] = True

    return recommendations


# 6️⃣ 간단한 시각화
def plot_audio_overview(y_segment, sr, band_energies, start_minutes=0):
    """
    오디오 개요 시각화 (파형 + 주파수 스펙트럼)
    """

    print(f"📈 오디오 개요 시각화")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 파형
    time_axis = np.linspace(0, len(y_segment)/sr, len(y_segment))
    axes[0].plot(time_axis, y_segment)
    axes[0].set_title(f'Waveform - {start_minutes:.1f}min segment')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 2. 주파수 대역별 에너지
    bands = list(band_energies.keys())
    energies = list(band_energies.values())

    bars = axes[1].bar(range(len(bands)), energies, color=[
                       'blue', 'orange', 'red', 'purple'])
    axes[1].set_title('Frequency Band Energy')
    axes[1].set_xticks(range(len(bands)))
    axes[1].set_xticklabels(bands, rotation=45, ha='right')
    axes[1].set_ylabel('Energy')

    # 막대 위에 값 표시
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{energy:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# 7️⃣ 전체 분석을 한 번에 (기존 기능)
def complete_audio_analysis(audio_input, sr=22050, start_minutes=0, end_minutes=None, duration_minutes=None):
    """
    전체 분석을 한 번에 실행 (기존 segment_noise_analysis_v2와 동일)
    """

    print("🚀 === 완전한 오디오 분석 ===")

    # 단계별 실행
    y_segment, sr, segment_info = extract_audio_segment(
        audio_input, sr, start_minutes, end_minutes, duration_minutes)
    stats = analyze_audio_stats(y_segment, sr)
    band_energies = analyze_frequency_bands(y_segment, sr)
    ratios = estimate_noise_type(band_energies)
    recommendations = recommend_noise_reduction(stats, ratios)

    # 시각화
    plot_audio_overview(y_segment, sr, band_energies, start_minutes)

    # 결과 통합
    result = {
        'segment_info': segment_info,
        'stats': stats,
        'band_energies': band_energies,
        'ratios': ratios,
        'recommendations': recommendations
    }

    print("✅ 전체 분석 완료!")
    return result


# 8️⃣ 사용 예시들
def usage_examples():
    """
    각 함수의 사용 예시
    """

    print("📚 === 사용법 예시 ===")
    print()
    print("# 1️⃣ 구간만 추출하고 싶을 때")
    print("y_segment, sr, info = extract_audio_segment('audio.wav', start_minutes=10, duration_minutes=5)")
    print()
    print("# 2️⃣ 기본 통계만 보고 싶을 때")
    print("stats = analyze_audio_stats(y_segment, sr)")
    print()
    print("# 3️⃣ 주파수 분석만 하고 싶을 때")
    print("band_energies = analyze_frequency_bands(y_segment, sr)")
    print()
    print("# 4️⃣ 노이즈 유형만 알고 싶을 때")
    print("ratios = estimate_noise_type(band_energies)")
    print()
    print("# 5️⃣ 권장사항만 보고 싶을 때")
    print("recommendations = recommend_noise_reduction(stats, ratios)")
    print()
    print("# 6️⃣ 간단한 시각화만 하고 싶을 때")
    print("plot_audio_overview(y_segment, sr, band_energies)")
    print()
    print("# 7️⃣ 전체 분석을 한 번에 하고 싶을 때")
    print("result = complete_audio_analysis('audio.wav', start_minutes=10, duration_minutes=5)")
    print()
    print("# 8️⃣ 단계별로 진행하고 싶을 때")
    print("y_segment, sr, info = extract_audio_segment('audio.wav', start_minutes=10, duration_minutes=5)")
    print("stats = analyze_audio_stats(y_segment, sr)")
    print("band_energies = analyze_frequency_bands(y_segment, sr)")
    print("# ... 필요한 단계만 선택적으로 실행")


if __name__ == "__main__":
    usage_examples()
