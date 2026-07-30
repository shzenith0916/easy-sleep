# -*- coding: utf-8 -*-
"""
clip_features.py
================
파이프라인 A(라벨 생성)와 B(모델 입력 전처리)를 잇는 공용 함수.

- episode_to_clips : A가 검출한 코골이 에피소드(start_sec, end_sec)를
                     모델 입력 단위인 고정 길이 클립들로 잘라낸다. (A→B 다리)
- clip_to_mel_db   : 클립 1개 → dB 멜스펙트로그램. **학습과 추론이 반드시
                     이 함수를 그대로 import 해서 공유**해야 train/inference
                     skew가 없다.

설계 메모
---------
- 멜 파라미터는 inference.py 와 동일하게 고정(N_FFT=2048, HOP_LENGTH=256,
  N_MELS=128). 기존 utils/audio_convert.py 의 convert_to_mel 은 이 인자들을
  받고도 librosa 에 넘기지 않아 hop=512(기본값)로 동작하는 버그가 있으니,
  새 코드는 반드시 이쪽(clip_to_mel_db)을 쓸 것.
- 모델 입력 전처리(확정): clip_to_logmel = DC offset 제거 → 멜 → dB(ref=np.max).
  RMS 정규화는 ref=np.max 때문에 결과가 수학적으로 동일(no-op)이라 제외했고,
  reduce_noise 는 1초 클립에서 degenerate(자기 자신을 노이즈로) 라 제외했다.
  디노이즈가 정말 필요하면 세그먼트 이전(긴 파일) 단계에서 진짜 조용한 구간을
  노이즈 프로파일로 써서 적용할 것.
"""

import numpy as np
import librosa

# 모델 입력 전처리 레시피 (inference.py 와 일치, 한 곳에서만 정의)
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128

# 모델이 한 번에 보는 오디오 길이(초). **학습과 추론이 반드시 동일해야 한다.**
# 코골이 호흡 주기(~2-6초)의 '드르렁-쉼-드르렁' 리듬을 한 창에 담도록 1초가 아닌
# 다중초로 둔다(권장 3~5초). 이 값 하나만 바꾸면 생성기·추론이 모두 따라온다.
# (멜은 logmel_to_image 에서 224x224 로 리사이즈되므로 길이가 달라도 입력 크기는 동일.
#  단, train/inference 가 다른 clip_sec 을 쓰면 '픽셀당 시간'이 달라져 skew 가 난다.)
CLIP_SEC = 4.0


def episode_to_clips(y, sr, seg, clip_sec=CLIP_SEC, hop_sec=None):
    """에피소드 구간을 고정 길이 클립들로 분할한다.

    Parameters
    ----------
    y : np.ndarray
        1차원 오디오 신호.
    sr : int
        샘플레이트.
    seg : tuple(float, float)
        (시작초, 끝초). snore_extract.py 가 내놓는 에피소드 형식과 동일.
    clip_sec : float
        클립 1개 길이(초). 모델이 먹는 단위. 기본 CLIP_SEC.
        **학습과 추론이 같은 값을 써야 한다.**
    hop_sec : float or None
        클립 간 이동 간격(초). None 이면 clip_sec(겹치지 않음).
        더 작으면 overlap(예: clip_sec/2 → 50% 겹침). hop 은 학습/추론이 달라도
        무방하다(각 창이 같은 시간폭을 의미하므로). 학습 샘플을 늘릴 땐 줄인다.

    Returns
    -------
    list[np.ndarray]
        각 원소가 길이 int(clip_sec*sr) 인 클립. 끝의 clip_sec 미만 잔여
        구간은 버린다(모델 입력 길이를 일정하게 유지하기 위함).
    """
    if hop_sec is None:
        hop_sec = clip_sec
    start = int(seg[0] * sr)
    end = int(seg[1] * sr)
    win = int(clip_sec * sr)
    hop = int(hop_sec * sr)
    if win <= 0 or hop <= 0:
        raise ValueError("clip_sec 와 hop_sec 는 0보다 커야 합니다.")

    return [y[s:s + win] for s in range(start, end - win + 1, hop)]


def clip_to_mel(clip, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """클립 1개 → power 멜스펙트로그램.

    멜 계산의 단일 소스. dB 가 필요하면 clip_to_mel_db 를 쓰고, power 멜이
    필요한 곳(예: audio_convert.convert_to_mel)은 이 함수를 위임해서 쓴다.

    Parameters
    ----------
    clip : np.ndarray
        1차원 오디오 클립.
    sr : int
        샘플레이트.
    n_fft, hop_length, n_mels : int
        멜 파라미터. 기본값은 모듈 상수(=inference.py 와 동일).

    Returns
    -------
    np.ndarray
        shape (n_mels, frames) 의 power 멜스펙트로그램.
    """
    return librosa.feature.melspectrogram(
        y=clip, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


def clip_to_mel_db(clip, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """클립 1개 → dB 스케일 멜스펙트로그램.

    학습과 추론이 이 함수를 공유해야 전처리 동일성이 보장된다.

    Parameters
    ----------
    clip : np.ndarray
        1차원 오디오 클립.
    sr : int
        샘플레이트.
    n_fft, hop_length, n_mels : int
        멜 파라미터. 기본값은 모듈 상수(=inference.py 와 동일).

    Returns
    -------
    np.ndarray
        shape (n_mels, frames) 의 dB 멜스펙트로그램.
    """
    mel = clip_to_mel(clip, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def remove_dc_offset(y):
    """신호에서 DC offset(상수 바이어스)을 제거한다."""
    return y - np.mean(y)


def clip_to_logmel(clip, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """모델 입력용 확정 전처리: DC offset 제거 → dB 멜스펙트로그램.

    **학습 데이터 생성과 추론이 이 함수를 그대로 공유**해야 train/inference
    skew가 없다. (RMS 정규화는 dB(ref=np.max) 때문에 no-op이라, reduce_noise는
    1초 클립에서 degenerate라 의도적으로 빼둠.)
    """
    return clip_to_mel_db(
        remove_dc_offset(clip), sr,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


def logmel_to_image(mel_db, img_size=(224, 224), top_db=80.0):
    """dB 멜스펙트로그램 → 모델 입력 이미지 (H, W, 3) uint8.

    결정적 변환(matplotlib/디스크 없음). **학습 이미지 생성기와 추론이 이
    함수를 그대로 공유**해야 한다. PNG 는 무손실이라, 생성기가 저장한 PNG 를
    다시 읽어도 여기서 바로 만든 배열과 픽셀이 동일하다.

    - dB 멜은 power_to_db(ref=np.max) 기준 대략 [-top_db, 0] 범위 → [0,1] 정규화
    - (n_mels, frames) 회색조를 img_size 로 리사이즈 후 3채널 복제
    - 0~255 uint8 로 반환(학습이 image_dataset_from_directory 로 읽는 0~255 와 정합)
    """
    from PIL import Image  # 무거운 의존이라 함수 안에서 import

    norm = np.clip((mel_db + top_db) / top_db, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    # PIL resize 는 (width, height) 순서
    img = Image.fromarray(u8).resize((img_size[1], img_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return np.stack([arr, arr, arr], axis=-1)
