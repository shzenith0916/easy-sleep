import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import librosa
import noisereduce as nr
from IPython.display import Audio, display
import soundfile as sf
from preprocessing.denoise import denoise_stationary, denoise_non_stationary


class AudioSegment:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio, self.sr = librosa.load(file_path, sr=16000)

    def extract_audio_segment(self, start_mins=0, duration_mins=5):
        """
        오디오에서 특정 시간대의 세그먼트를 추출하는 함수

        Args:
            audio (numpy.ndarray): 오디오 데이터
            sr (int): 샘플링 레이트
            start_minute (int): 시작 시간 (분)
            duration_minutes (int): 세그먼트 길이 (분)

        Returns:
            추출된 세그먼트 오디오 데이터 (numpy.ndarray)
        """
        audio, sr = self.audio, self.sr
        start_sample = int(start_mins * 60 * sr)
        end_sample = start_sample + int(duration_mins * 60 * sr)

        # 샘플 오디오 길이 확인하여, 요청한 시간이 오디오파일 길이 보다 길면, 오디오 파일 길이로 조정
        if end_sample > len(audio):
            print(
                f"Warning: The requested sample lenght is longer than the audio file. {len(audio)/sr/60:.1f}분까지 추출출")
            end_sample = len(audio)
            segment = audio[start_sample:end_sample]
            print(f"Segment Extraction: {start_mins}분 ~ {end_sample/sr:.1f}분")
            print(f"실제 길이: {len(segment)/sr:.1f}초")
        else:
            segment = audio[start_sample:end_sample]
            print(
                f"Segment Extraction: {start_mins}분 ~ {start_mins + duration_mins}분")
            print(f"실제 길이: {len(segment)/sr:.1f}초")

        return segment 


    def load_and_extract_segment(self, file_path, start_mins=0, duration_mins=5):
        """
        오디오 파일을 로드하고 특정 세그먼트를 추출하는 함수

        Args:
            audio_path: 오디오 파일 경로
            start_minute: 시작 시간 (분)
            duration_minutes: 세그먼트 길이 (분)
            sr: 샘플링 레이트

        Returns:
            세그먼트 오디오 데이터와 샘플링 레이트
        """
        # 전체 오디오 로드
        audio, sr = librosa.load(file_path, sr=16000)
        print(f"전체 오디오 길이: {len(audio)/sr/60:.1f}분")

        # 세그먼트 추출
        segment = self.extract_audio_segment(
            audio, sr, start_mins, duration_mins)

        return segment, sr


class AudioPlay:
    def __init__(self, original_segment, sr=16000):
        self.original_segment = original_segment
        self.sr = sr

    def display_audio_segment(self, original_segment, sr, title="Audio Segment, 분할된 오디오"):
        """오디오 세그먼트를 재생할 수 있도록 하는 함수"""

        try:
            print(f"{title}")
            print(f"길이: {len(original_segment)/sr:.1f}초")
            print(f"샘플링 레이트: {sr}Hz")

            # 오디오 재생
            display(Audio(original_segment, rate=sr))

        except ImportError:
            print("IPython.display를 사용할 수 없습니다.")

        return original_segment, sr

    def save_segment(self, original_segment, sr, output_path, title="Audio Segment"):
        """
        오디오 세그먼트를 파일로 저장하는 함수

        Args:
            original_segment: 오리지널 분할된 오디오 데이터
            sr: 샘플링 레이트
            output_path: 저장할 파일 경로
            title: 제목
        """
        try:
            sf.write(output_path, self.original_segment, sr)
            print(f"{title} 저장 완료: {output_path}")
            print(f"길이: {len(original_segment)/sr:.1f}초")

        except ImportError:
            print("soundfile 라이브러리가 필요합니다. pip install soundfile")
            return False

        return True

    def compare_original_vs_denoised(self, original_segment, denoised_segment, sr=16000, denoise_method="stationary"):
        """
        원본과 노이즈 제거된 세그먼트를 비교하는 함수

        Args:
            original_segment: 원본 세그먼트
            denoised_segment: 노이즈 제거된 세그먼트
            sr: 샘플링 레이트 (기본값: 16000)
            denoise_method: 노이즈 제거 방법 (stationary, non_stationary)
        """

        print("=== 원본 vs 노이즈 제거 비교 ===")

        # 원본 세그먼트
        self.display_audio_segment(
            original_segment, sr, f"원본 오디오)")

        # 노이즈 제거된 세그먼트
        if denoise_method == "stationary":
            self.display_audio_segment(denoised_segment, sr,
                                       f"Stationary 노이즈 제거)")
        else:
            self.display_audio_segment(
                denoised_segment, sr, f"Non-stationary 노이즈 제거)")

        return original_segment, denoised_segment, sr


def extract_from_processed_audio(processed_result, start_mins=0, duration_mins=5):
    """
    노이즈 제거 후 처리된 오디오에서 세그먼트 추출
    
    Parameters:
    processed_result: (audio, sr) 튜플 또는 오디오 데이터
    start_mins: 시작 시간 (분)
    duration_mins: 지속 시간 (분)
    
    Returns:
    segment: 추출된 오디오 세그먼트
    """
    if isinstance(processed_result, tuple):
        audio, sr = processed_result
    else:
        audio, sr = processed_result, 16000  # 기본 sr

    start_sample = int(start_mins * 60 * sr)
    end_sample = start_sample + int(duration_mins * 60 * sr)
    end_sample = min(end_sample, len(audio))

    segment = audio[start_sample:end_sample]

    print(f"Segment Extraction: {start_mins}분 ~ {start_mins + duration_mins}분")
    print(f"실제 길이: {len(segment)/sr:.1f}초")

    return segment



# 사용 예시
if __name__ == "__main__":

    audio_file = r"c:\\Users\\USER\\Documents\\코골이\\easy_sleep\\PSG_sample.wav"

    # AudioSegment 클래스 인스턴스 생성
    audio_segment = AudioSegment(audio_file)
    # 5분 세그먼트 추출 -> Original Segment
    original_segment = audio_segment.extract_audio_segment(
        start_mins=0, duration_mins=5)

    denoised_stationary_segment = denoise_stationary(
        original_segment, sr=16000, segment_mins=5)
    denoised_non_stationary_segment = denoise_non_stationary(
        original_segment, sr=16000, window_mins=5)

    # AudioPlay 클래스 인스턴스 생성
    audio_play = AudioPlay(original_segment)

    # 원본과 stationary노이즈 제거 비교
    original, denoised, sr = audio_play.compare_original_vs_denoised(
        original_segment, denoised_stationary_segment, denoise_method="stationary")

    # 원본과 non-stationary노이즈 제거 비교
    original, denoised, sr = audio_play.compare_original_vs_denoised(
        original_segment, denoised_non_stationary_segment, denoise_method="non_stationary")
