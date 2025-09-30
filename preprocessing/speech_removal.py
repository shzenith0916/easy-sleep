import librosa
import numpy as np
import matplotlib.pyplot as plt
import pydub
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import datetime
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import os
from utils.audio_convert import Audio_Converter


class SpeechRemover:
    def __init__(self, sample_rate=16000, frame_length=1024, hop_length=512):
        """
        말소리 제거 클래스 초기화

        매개변수:
            sample_rate: 샘플링 레이트 (기본값: 16000Hz)
            frame_length: 프레임 길이 (기본값: 1024)
            hop_length: 홉 길이 (기본값: 512)
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def detect_speech_segments(self, audio, sr, energy_threshold=0.01,
                               spectral_centroid_threshold=1000,
                               zero_crossing_threshold=0.1):
        """
        음성 활동 감지(VAD)를 통해 말소리 구간을 찾습니다.

        매개변수: 
            audio: 오디오 신호 (샘플링레이트 만큼의 길이의 배열 형태) 
            sr: 샘플링 레이트
            energy_threshold: 에너지 임계값
            spectral_centroid_threshold: 스펙트럼 중심 임계값
            zero_crossing_threshold: 제로 크로싱 임계값

        반환:
            말소리 구간 리스트 [(시작_초, 종료_초), ...]
        """
        # 프레임 단위로 분석
        frame_length_samples = int(self.frame_length)
        hop_length_samples = int(self.hop_length)

        speech_segments = []
        current_segment_start = None

        for i in range(0, len(audio) - frame_length_samples, hop_length_samples):
            frame = audio[i:i + frame_length_samples]
            time_sec = i / sr

            # 1. 에너지 기반 감지
            energy = np.mean(frame ** 2)

            # 2. 스펙트럼 중심 기반 감지 (말소리는 높은 주파수 성분이 많음)
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=frame, sr=sr))

            # 3. 제로 크로싱 비율 기반 감지 (말소리는 복잡한 파형)
            zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
            zcr = zero_crossings / len(frame)

            # 말소리 판정 조건
            is_speech = (energy > energy_threshold and
                         spectral_centroid > spectral_centroid_threshold and
                         zcr > zero_crossing_threshold)

            if is_speech and current_segment_start is None:
                # 말소리 구간 시작
                current_segment_start = time_sec
            elif not is_speech and current_segment_start is not None:
                # 말소리 구간 종료
                current_segment_end = time_sec
                speech_segments.append(
                    (current_segment_start, current_segment_end))
                current_segment_start = None

        # 마지막 구간 처리
        if current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio) / sr))

        return speech_segments

    def remove_speech_segments(self, audio, sr, speech_segments,
                               silence_duration=0.1, fade_duration=0.05):
        """
        말소리 구간을 무음으로 대체합니다.

        매개변수:
            audio: 원본 오디오 신호
            sr: 샘플링 레이트
            speech_segments: 말소리 구간 리스트
            silence_duration: 무음 구간 앞뒤로 추가할 페이드 시간
            fade_duration: 페이드 인/아웃 시간

        반환:
            말소리가 제거된 오디오 신호
        """
        processed_audio = audio.copy()

        for start_sec, end_sec in speech_segments:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # 페이드 아웃 적용
            fade_samples = int(fade_duration * sr)
            if start_sample > fade_samples:
                fade_out = np.linspace(1, 0, fade_samples)
                processed_audio[start_sample -
                                fade_samples:start_sample] *= fade_out

            # 페이드 인 적용
            if end_sample + fade_samples < len(processed_audio):
                fade_in = np.linspace(0, 1, fade_samples)
                processed_audio[end_sample:end_sample+fade_samples] *= fade_in

            # 말소리 구간을 무음으로 대체
            processed_audio[start_sample:end_sample] = 0

        return processed_audio

    def detect_breathing_sounds(self, audio, sr, low_freq_threshold=200,
                                high_freq_threshold=2000):
        """
        호흡 소리를 감지합니다.

        매개변수:
            audio: 오디오 신호
            sr: 샘플링 레이트
            low_freq_threshold: 저주파 임계값
            high_freq_threshold: 고주파 임계값

        반환:
            호흡 소리 구간 리스트
        """
        # 저주파 필터링 (호흡 소리는 주로 저주파)
        sos = signal.butter(4, low_freq_threshold,
                            btype='high', fs=sr, output='sos')
        filtered_audio = signal.sosfilt(sos, audio)

        # 에너지 기반 호흡 소리 감지
        frame_length_samples = int(self.frame_length)
        hop_length_samples = int(self.hop_length)

        breathing_segments = []
        current_segment_start = None

        for i in range(0, len(filtered_audio) - frame_length_samples, hop_length_samples):
            frame = filtered_audio[i:i + frame_length_samples]
            time_sec = i / sr

            # 에너지 계산
            energy = np.mean(frame ** 2)

            # 호흡 소리 임계값 (말소리보다 낮음)
            breathing_threshold = 0.005

            if energy > breathing_threshold and current_segment_start is None:
                current_segment_start = time_sec
            elif energy <= breathing_threshold and current_segment_start is not None:
                current_segment_end = time_sec
                # 최소 길이 필터링 (너무 짧은 구간 제거)
                if current_segment_end - current_segment_start > 0.5:
                    breathing_segments.append(
                        (current_segment_start, current_segment_end))
                current_segment_start = None

        return breathing_segments

    def process_audio(self, audio_path, output_path=None):
        """
        오디오에서 말소리를 제거하고 코골이/호흡 소리만 남깁니다.

        매개변수:
            audio_path: 입력 오디오 파일 경로
            output_path: 출력 파일 경로 (선택사항)

        반환:
            처리된 오디오 신호와 구간 정보
        """
        # 오디오 로드
        audio, sr = self.Audio_Converter.load_audio_and_resample(audio_path)
        if audio is None:
            return None, None

        print(f"오디오 로드 완료: {len(audio)/sr:.2f}초")

        # 말소리 구간 감지
        speech_segments = self.detect_speech_segments(audio, sr)
        print(f"감지된 말소리 구간 수: {len(speech_segments)}")

        # 말소리 제거
        processed_audio = self.remove_speech_segments(
            audio, sr, speech_segments)
        print("말소리 제거 완료")

        # 호흡 소리 감지
        breathing_segments = self.detect_breathing_sounds(processed_audio, sr)
        print(f"감지된 호흡 소리 구간 수: {len(breathing_segments)}")

        # 결과 저장
        if output_path:
            librosa.output.write_wav(output_path, processed_audio, sr)
            print(f"처리된 오디오 저장: {output_path}")

        # 구간 정보를 DataFrame으로 변환
        segments_info = []

        # 말소리 구간 정보
        for i, (start, end) in enumerate(speech_segments):
            segments_info.append({
                "구간_타입": "말소리",
                "구간_번호": i + 1,
                "시작_시간": str(datetime.timedelta(seconds=round(start, 1))),
                "종료_시간": str(datetime.timedelta(seconds=round(end, 1))),
                "시작_초": start,
                "종료_초": end,
                "구간_길이": end - start
            })

        # 호흡 소리 구간 정보
        for i, (start, end) in enumerate(breathing_segments):
            segments_info.append({
                "구간_타입": "호흡소리",
                "구간_번호": i + 1,
                "시작_시간": str(datetime.timedelta(seconds=round(start, 1))),
                "종료_시간": str(datetime.timedelta(seconds=round(end, 1))),
                "시작_초": start,
                "종료_초": end,
                "구간_길이": end - start
            })

        segments_df = pd.DataFrame(segments_info)

        return processed_audio, segments_df


def visualize_segments(audio, sr, segments_df, output_path=None):
    """
    감지된 구간들을 시각화합니다.

    매개변수:
        audio: 오디오 신호
        sr: 샘플링 레이트
        segments_df: 구간 정보 DataFrame
        output_path: 저장할 이미지 경로 (선택사항)
    """
    plt.figure(figsize=(15, 6))

    # 파형 그리기
    time_axis = np.linspace(0, len(audio)/sr, len(audio))
    plt.plot(time_axis, audio, alpha=0.7, color='blue', linewidth=0.5)

    # 구간별 색상 표시
    colors = {'말소리': 'red', '호흡소리': 'green'}

    for _, row in segments_df.iterrows():
        start_sec = row['시작_초']
        end_sec = row['종료_초']
        segment_type = row['구간_타입']

        plt.axvspan(start_sec, end_sec, alpha=0.3,
                    color=colors.get(segment_type, 'gray'),
                    label=segment_type if segment_type not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('시간 (초)')
    plt.ylabel('진폭')
    plt.title('말소리 제거 및 호흡소리 감지 결과')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"시각화 결과 저장: {output_path}")

    plt.show()


# 실행 코드
if __name__ == "__main__":
    # 오디오 파일 경로 (프로젝트 내 파일로 변경)
    audio_path = r"C:\Users\USER\Documents\코골이\easy_sleep\data\raw\20230904_SnoringSample_1.wav"
    output_dir = r"C:\Users\USER\Documents\코골이\easy_sleep\data\processed"

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 말소리 제거기 초기화
        remover = SpeechRemover(sample_rate=16000)

        # 오디오 처리
        output_path = os.path.join(output_dir, "speech_removed_audio.wav")
        processed_audio, segments_df = remover.process_audio(
            audio_path, output_path)

        if processed_audio is not None and segments_df is not None:
            print("\n=== 처리 결과 ===")
            print(segments_df)

            # 통계 정보
            speech_stats = segments_df[segments_df['구간_타입'] == '말소리']['구간_길이']
            breathing_stats = segments_df[segments_df['구간_타입']
                                          == '호흡소리']['구간_길이']

            print(f"\n말소리 구간 통계:")
            print(f"  - 총 구간 수: {len(speech_stats)}")
            print(f"  - 총 길이: {speech_stats.sum():.2f}초")
            print(f"  - 평균 길이: {speech_stats.mean():.2f}초")

            print(f"\n호흡소리 구간 통계:")
            print(f"  - 총 구간 수: {len(breathing_stats)}")
            print(f"  - 총 길이: {breathing_stats.sum():.2f}초")
            print(f"  - 평균 길이: {breathing_stats.mean():.2f}초")

            # 결과 저장
            segments_df.to_csv(os.path.join(output_dir, "segments_info.csv"),
                               index=False, encoding='utf-8-sig')

            # 시각화
            viz_output_path = os.path.join(
                output_dir, "segments_visualization.png")
            visualize_segments(processed_audio, 16000,
                               segments_df, viz_output_path)

            print(f"\n모든 결과가 {output_dir}에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
