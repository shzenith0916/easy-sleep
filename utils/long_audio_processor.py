import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
import os


class LongAudioProcessor:
    """
    장시간 오디오 처리를 위한 클래스
    """

    def __init__(self, sr=16000):
        self.sr = sr

    def process_long_audio(self, audio, sr=None, segment_mins=30,
                           noise_reduction_method='adaptive',
                           prop_decrease=0.7):
        """
        장시간 오디오를 구간별로 처리

        Parameters:
        - audio: 오디오 데이터 또는 파일 경로
        - sr: 샘플링 레이트
        - segment_mins: 구간 길이 (분)
        - noise_reduction_method: 'adaptive', 'stationary', 'none'
        - prop_decrease: 노이즈 제거 강도 (0.0-1.0)
        """
        if sr is None:
            sr = self.sr

        # 오디오 로드
        if isinstance(audio, str):
            y, sr = librosa.load(audio, sr=sr)
        else:
            y = audio

        print(f"🎵 오디오 길이: {len(y)/sr/3600:.1f}시간")
        print(f"📊 구간별 처리: {segment_mins}분씩")

        segment_length = segment_mins * 60 * sr
        processed_segments = []

        total_segments = len(y) // segment_length + 1
        print(f"📈 총 {total_segments}개 구간 처리 예정")

        for i in range(0, len(y), segment_length):
            segment = y[i: i + segment_length]
            segment_num = i // segment_length + 1

            print(f"🔄 구간 {segment_num}/{total_segments} 처리 중...")

            # 노이즈 제거 방법 선택
            if noise_reduction_method == 'adaptive':
                # 적응적 노이즈 제거 (조용한 구간 찾기)
                denoised_segment = self._adaptive_denoise(segment, sr)
            elif noise_reduction_method == 'stationary':
                # 정상 노이즈 제거
                denoised_segment = nr.reduce_noise(
                    y=segment, sr=sr, stationary=True, prop_decrease=prop_decrease
                )
            elif noise_reduction_method == 'none':
                # 노이즈 제거 없음
                denoised_segment = segment
            else:
                raise ValueError(
                    "noise_reduction_method는 'adaptive', 'stationary', 'none' 중 하나여야 합니다.")

            processed_segments.append(denoised_segment)

        # 모든 구간 합치기
        final_audio = np.concatenate(processed_segments)
        print(f"✅ 처리 완료! 최종 길이: {len(final_audio)/sr/3600:.1f}시간")

        return final_audio, sr

    def _adaptive_denoise(self, segment, sr, noise_duration_sec=20):
        """
        적응적 노이즈 제거 (조용한 구간을 노이즈 프로파일로 사용)
        """
        # 조용한 구간 찾기 (처음 20초)
        noise_sample = segment[:noise_duration_sec * sr]

        # 노이즈 제거
        return nr.reduce_noise(
            y=segment,
            sr=sr,
            y_noise=noise_sample,
            stationary=False,
            prop_decrease=0.7
        )

    def save_processed_audio(self, audio, sr, output_path):
        """
        처리된 오디오를 파일로 저장
        """
        import soundfile as sf

        sf.write(output_path, audio, sr)
        duration_hours = len(audio) / sr / 3600
        print(f"💾 저장 완료: {output_path} ({duration_hours:.1f}시간)")


# 사용 예시 함수들
def process_sleep_audio(audio_file, output_file=None, segment_mins=30,
                        noise_method='adaptive', sr=16000):
    """
    수면 오디오 처리 (간단한 사용법)

    Parameters:
    - audio_file: 입력 오디오 파일
    - output_file: 출력 파일 (None이면 자동 생성)
    - segment_mins: 구간 길이 (분)
    - noise_method: 노이즈 제거 방법
    - sr: 샘플링 레이트
    """
    processor = LongAudioProcessor(sr=sr)

    # 오디오 처리
    processed_audio, sr = processor.process_long_audio(
        audio=audio_file,
        sr=sr,
        segment_mins=segment_mins,
        noise_reduction_method=noise_method
    )

    # 파일 저장
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = f"{base_name}_processed.wav"

    processor.save_processed_audio(processed_audio, sr, output_file)

    return processed_audio, sr


def compare_processing_methods(audio_file, sr=16000, segment_mins=10):
    """
    다양한 처리 방법 비교 (테스트용)
    """
    processor = LongAudioProcessor(sr=sr)

    methods = {
        '원본': None,
        '적응적 노이즈 제거': 'adaptive',
        '정상 노이즈 제거': 'stationary'
    }

    results = {}

    for method_name, method in methods.items():
        print(f"\n🔍 {method_name} 처리 중...")

        if method is None:
            # 원본
            audio, sr = librosa.load(audio_file, sr=sr)
            results[method_name] = audio
        else:
            # 노이즈 제거
            processed_audio, sr = processor.process_long_audio(
                audio=audio_file,
                sr=sr,
                segment_mins=segment_mins,
                noise_reduction_method=method
            )
            results[method_name] = processed_audio

    return results, sr


# 사용법 예시
if __name__ == "__main__":
    print("🎵 장시간 오디오 처리기 사용법:")
    print("1. process_sleep_audio('audio.wav') - 간단한 처리")
    print("2. compare_processing_methods('audio.wav') - 방법 비교")
    print("3. processor = LongAudioProcessor(sr=16000) - 개별 사용")

