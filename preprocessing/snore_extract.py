#!/usr/bin/env python3
"""
snore_extract.py
================
규칙기반 코골이 구간 검출 → 구간별 클립 파일 추출.

설계 철학
---------
- 파일 종류(6시간/15분/이미잘린것)로 분기하지 않는다.
  모든 파일에 동일 파이프라인을 적용하고, 임계값은 "파일 내부 통계"에서
  적응적으로 뽑는다. 따라서 train 전처리 == inference 전처리가 자동 보장된다.
- 학습데이터 생성용이므로 recall보다 PRECISION 우선.
  애매하면 버린다(보수적). 잘린 클립은 확실히 코골이여야 라벨이 깨끗하다.

검출 신호 (3가지를 AND로 결합 → 단발 노이즈/잡음 배제)
  1) 적응형 에너지   : 파일별 노이즈 플로어(하위 백분위) 대비 충분히 큰가
  2) 저주파 비중     : 코골이 에너지는 대략 80~600Hz에 몰린다
  3) 호흡 주기성     : 검출 프레임들이 규칙적 간격(약 2~6초)으로 반복되는가

사용법
------
  python snore_extract.py INPUT [INPUT ...] [옵션]
  python snore_extract.py night.wav
  python snore_extract.py ./recordings/*.m4a -o ./clips
  python snore_extract.py night.wav --sensitivity high --dry-run

옵션은 -h 로 확인.

의존성: librosa, soundfile, numpy, scipy
  pip install librosa soundfile numpy scipy
"""

import argparse
import os
import sys
import glob
import csv

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


# ----------------------------------------------------------------------------
# 설정값 (대부분 적응형이라 건드릴 일은 적음)
# ----------------------------------------------------------------------------
SR = 16000                 # 분석용 샘플레이트로 통일 (코골이는 고주파 불필요)
FRAME_SEC = 1.0            # 분석 윈도우 길이(초)
HOP_SEC = 0.5             # 윈도우 이동 간격(초) → 50% overlap

SNORE_BAND = (80, 600)     # 코골이 핵심 주파수 대역 (Hz)

# 민감도 프리셋: (에너지 백분위 baseline, baseline 대비 배수, 저주파비중 최소)
SENSITIVITY = {
    # precision 최우선 - 확실한 것만
    "high":   {"floor_pct": 30, "energy_mult": 4.0, "lowband_min": 0.55},
    # 균형
    "medium": {"floor_pct": 25, "energy_mult": 3.0, "lowband_min": 0.45},
    # recall 우선 - 더 많이 잡되 노이즈도 늘어남
    "low":    {"floor_pct": 20, "energy_mult": 2.2, "lowband_min": 0.35},
}

# 세그먼트 후처리
# 코골이는 호흡 주기(~2-6초)마다 끊겼다 이어진다. 개별 호흡을 하나의
# "코골이 에피소드"로 묶으려면 병합 간격이 호흡 주기보다 커야 한다.
MERGE_GAP_SEC = 8.0        # 이 간격 이내 검출은 한 에피소드로 병합
MIN_SEG_SEC = 5.0          # 이보다 짧은 에피소드는 버림(우연한 단발 소음 제거)
PAD_SEC = 0.3              # 클립 앞뒤 여유

# 호흡 주기성 검사
PERIODICITY_MIN_EVENTS = 3  # 한 세그먼트 안에 최소 이만큼 반복 이벤트가 있어야 인정
PERIOD_RANGE_SEC = (1.5, 7.0)  # 정상 코골이 호흡 간격 범위


def bandpass_sos(low, high, sr, order=4):
    nyq = sr / 2
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def frame_features(y, sr):
    """프레임별 (전체에너지, 저주파대역에너지비중) 반환."""
    frame_len = int(FRAME_SEC * sr)
    hop_len = int(HOP_SEC * sr)

    # 전체 RMS 에너지 (프레임별)
    rms = librosa.feature.rms(
        y=y, frame_length=frame_len, hop_length=hop_len)[0]

    # 저주파 대역만 통과시킨 신호의 RMS
    sos = bandpass_sos(SNORE_BAND[0], SNORE_BAND[1], sr)
    y_band = sosfiltfilt(sos, y)
    rms_band = librosa.feature.rms(
        y=y_band, frame_length=frame_len, hop_length=hop_len
    )[0]

    # 길이 맞추기
    n = min(len(rms), len(rms_band))
    rms, rms_band = rms[:n], rms_band[:n]

    eps = 1e-10
    lowband_ratio = rms_band / (rms + eps)

    times = librosa.frames_to_time(
        np.arange(n), sr=sr, hop_length=hop_len
    )
    return times, rms, lowband_ratio


def detect_frames(rms, lowband_ratio, cfg):
    """프레임별 코골이 후보 boolean 마스크."""
    # 파일별 적응형 노이즈 플로어
    floor = np.percentile(rms, cfg["floor_pct"])
    energy_thr = floor * cfg["energy_mult"]

    is_loud = rms > energy_thr
    is_lowfreq = lowband_ratio > cfg["lowband_min"]
    return is_loud & is_lowfreq


def group_segments(mask, times):
    """boolean 마스크 → (start_sec, end_sec) 세그먼트 리스트. 인접 병합 포함."""
    if not mask.any():
        return []

    hop = HOP_SEC
    idx = np.where(mask)[0]

    segs = []
    seg_start = times[idx[0]]
    prev = times[idx[0]]
    for t in times[idx[1:]]:
        if t - prev <= MERGE_GAP_SEC:
            prev = t
        else:
            segs.append([seg_start, prev + FRAME_SEC])
            seg_start = t
            prev = t
    segs.append([seg_start, prev + FRAME_SEC])
    return segs


def check_periodicity(seg, mask, times):
    """세그먼트가 호흡 주기성을 보이는지 검사 (단발 소음 vs 반복 코골이 구분)."""
    s, e = seg
    in_seg = (times >= s) & (times < e)
    local = mask & in_seg
    if local.sum() < PERIODICITY_MIN_EVENTS:
        # 너무 짧으면 주기성 판단 불가 → 보수적으로 통과시키되
        # MIN_SEG_SEC 필터가 어차피 짧은건 거른다
        return True

    # 검출 프레임의 시간 → 연속 덩어리(이벤트)로 묶고, 각 이벤트의 시작점 추출.
    # "이벤트 사이 침묵(off 구간)"이 호흡 주기에 해당한다.
    det_times = times[local]
    event_starts = [det_times[0]]
    for prev, cur in zip(det_times[:-1], det_times[1:]):
        # 검출이 끊겼다가(off) 다시 시작되면 새 호흡 이벤트
        if cur - prev > HOP_SEC * 1.5:
            event_starts.append(cur)

    if len(event_starts) < PERIODICITY_MIN_EVENTS:
        return False

    gaps = np.diff(event_starts)
    # 이벤트 시작 간격(호흡 주기) 대부분이 정상 범위 안인가
    good = np.mean(
        (gaps >= PERIOD_RANGE_SEC[0]) & (gaps <= PERIOD_RANGE_SEC[1])
    )
    return good >= 0.5


def process_file(path, out_dir, cfg, dry_run, csv_writer):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n[+] {os.path.basename(path)}")

    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as ex:
        print(f"    ! 로드 실패: {ex}")
        return 0

    dur = len(y) / SR
    print(f"    길이 {dur/60:.1f}분, 분석 중...")

    times, rms, lowband_ratio = frame_features(y, SR)
    mask = detect_frames(rms, lowband_ratio, cfg)
    segs = group_segments(mask, times)

    # 필터: 최소 길이 + 주기성
    kept = []
    for seg in segs:
        if seg[1] - seg[0] < MIN_SEG_SEC:
            continue
        if not check_periodicity(seg, mask, times):
            continue
        kept.append(seg)

    print(f"    후보 {len(segs)}개 → 필터 후 {len(kept)}개 세그먼트")

    if not kept:
        return 0

    # 원본을 원래 SR로 다시 로드해 품질 보존하며 자름
    y_orig, sr_orig = librosa.load(path, sr=None, mono=True)

    count = 0
    for i, (s, e) in enumerate(kept, 1):
        s_pad = max(0, s - PAD_SEC)
        e_pad = min(dur, e + PAD_SEC)
        clip_name = f"{name}_snore_{i:03d}_{int(s_pad)}s-{int(e_pad)}s.wav"

        if csv_writer:
            csv_writer.writerow([
                os.path.basename(path), i,
                f"{s_pad:.2f}", f"{e_pad:.2f}", f"{e_pad - s_pad:.2f}"
            ])

        if dry_run:
            count += 1
            continue

        a = int(s_pad * sr_orig)
        b = int(e_pad * sr_orig)
        sf.write(os.path.join(out_dir, clip_name), y_orig[a:b], sr_orig)
        count += 1

    total = sum(e - s for s, e in kept)
    print(f"    → {count}개 클립, 총 {total/60:.1f}분"
          + (" (dry-run, 파일 미생성)" if dry_run else ""))
    return count


def main():
    ap = argparse.ArgumentParser(
        description="규칙기반 코골이 구간 추출기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+",
                    help="오디오 파일 경로 (와일드카드 가능: *.wav)")
    ap.add_argument("-o", "--out", default="./snore_clips",
                    help="출력 폴더 (기본: ./snore_clips)")
    ap.add_argument("-s", "--sensitivity", default="high",
                    choices=["high", "medium", "low"],
                    help="민감도. high=정밀도우선(기본), low=재현율우선")
    ap.add_argument("--dry-run", action="store_true",
                    help="실제 파일 생성 없이 검출 결과만 출력")
    ap.add_argument("--csv", default=None,
                    help="검출 타임스탬프를 CSV로 저장할 경로")
    args = ap.parse_args()

    # 입력 확장 (셸이 와일드카드 처리 안 한 경우 대비)
    files = []
    for p in args.inputs:
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("입력 파일을 찾을 수 없습니다.")
        sys.exit(1)

    cfg = SENSITIVITY[args.sensitivity]
    print(f"민감도={args.sensitivity}  대상 {len(files)}개 파일")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    csv_file = csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["source", "idx", "start_sec", "end_sec", "dur_sec"])

    total = 0
    for f in files:
        total += process_file(f, args.out, cfg, args.dry_run, csv_writer)

    if csv_file:
        csv_file.close()
        print(f"\nCSV 저장: {args.csv}")

    print(f"\n완료. 총 {total}개 클립"
          + ("" if args.dry_run else f" → {args.out}/"))


if __name__ == "__main__":
    main()
