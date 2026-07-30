#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_training_images.py
========================
오디오 클립 → 학습용 멜 이미지 생성기.

학습(model_train)은 image_dataset_from_directory 로 다음 구조의 이미지를 읽는다:
    <out>/1/*.png   (snore)
    <out>/0/*.png   (no-snore)

이 스크립트는 wav 클립들을 1초 윈도우로 쪼개고, **추론(inference.py)과 동일한**
clip_to_logmel + logmel_to_image 를 써서 그 폴더를 채운다. 같은 함수를 공유하므로
train == inference 전처리가 자동 보장된다.

전형적 흐름
-----------
1) snore_extract.py 로 코골이 클립 추출:
       python preprocessing/snore_extract.py ./recordings/*.m4a -o ./snore_clips
2) 코골이 클립 → 양성 이미지, 비코골이 클립 → 음성 이미지:
       python preprocessing/build_training_images.py \
           --snore ./snore_clips --nosnore ./nosnore_clips -o ./snoring_data_process

의존성: librosa, numpy, pillow  (clip_features 가 utils/ 에 있어야 함)
"""

import os
import sys
import glob
import argparse

import librosa
from PIL import Image

# utils/ 의 clip_features 를 import (추론과 동일한 전처리 함수 공유)
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from clip_features import (episode_to_clips, clip_to_logmel, logmel_to_image,
                           N_FFT, HOP_LENGTH, N_MELS, CLIP_SEC)

SR = 16000
HOP_SEC = CLIP_SEC  # 기본은 비겹침. 학습 샘플을 늘리려면 --hop-sec 로 줄여 overlap.
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def wav_to_clip_images(wav_path, out_dir, sr=SR, clip_sec=CLIP_SEC, hop_sec=HOP_SEC):
    """wav 1개 → 1초 클립 이미지들 저장. 저장한 이미지 수를 반환."""
    y, _ = librosa.load(wav_path, sr=sr)
    clips = episode_to_clips(y, sr, (0.0, len(y) / sr), clip_sec=clip_sec, hop_sec=hop_sec)

    stem = os.path.splitext(os.path.basename(wav_path))[0]
    saved = 0
    for i, clip in enumerate(clips):
        mel_db = clip_to_logmel(clip, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        img = logmel_to_image(mel_db)
        Image.fromarray(img).save(os.path.join(out_dir, f"{stem}_{i:03d}.png"))
        saved += 1
    return saved


def build_class(src_dir, out_dir, **kw):
    """src_dir 의 모든 오디오 → out_dir 에 이미지. (out_dir = <out>/0 또는 <out>/1)"""
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in glob.glob(os.path.join(src_dir, "*"))
             if f.lower().endswith(AUDIO_EXTS)]
    if not files:
        print(f"    ! {src_dir} 에 오디오 파일이 없습니다.")
        return 0

    total = 0
    for f in files:
        n = wav_to_clip_images(f, out_dir, **kw)
        total += n
        print(f"    {os.path.basename(f)} → {n}개 클립 이미지")
    return total


def main():
    ap = argparse.ArgumentParser(
        description="오디오 클립 → 학습용 멜 이미지 (추론과 동일 전처리)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snore", help="코골이(양성, label 1) 클립 폴더")
    ap.add_argument("--nosnore", help="비코골이(음성, label 0) 클립 폴더")
    ap.add_argument("-o", "--out", default="./snoring_data_process",
                    help="출력 루트 (기본: ./snoring_data_process). 하위에 0/ 1/ 생성")
    ap.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    ap.add_argument("--hop-sec", type=float, default=HOP_SEC)
    args = ap.parse_args()

    if not args.snore and not args.nosnore:
        ap.error("--snore 또는 --nosnore 중 최소 하나는 지정해야 합니다.")

    kw = dict(clip_sec=args.clip_sec, hop_sec=args.hop_sec)
    grand = 0
    if args.snore:
        print(f"[snore → {args.out}/1]")
        grand += build_class(args.snore, os.path.join(args.out, "1"), **kw)
    if args.nosnore:
        print(f"[no-snore → {args.out}/0]")
        grand += build_class(args.nosnore, os.path.join(args.out, "0"), **kw)

    print(f"\n완료. 총 {grand}개 이미지 → {args.out}/")


if __name__ == "__main__":
    main()
