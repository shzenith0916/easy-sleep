# 모델 학습 시도 정리 — 왜 코드를 지웠고, 무엇을 남겼나

2024-11 에 시도한 CNN 학습 코드(`training/`, `inference/`, `notebooks/train_test2.ipynb`)를
**삭제**했다. 이 문서는 그 판단 근거와, 재학습 때 반드시 알고 있어야 할 것을 남긴다.

> 관련 문서: [EDA.md](EDA.md) (전처리 탐색 결론) · [pipeline_overview.ipynb](pipeline_overview.ipynb)
> (A=라벨생성 / B=모델입력 분리) · [../preprocessing/snore_extract_explanation.md](../preprocessing/snore_extract_explanation.md)

---

## 1. 왜 지웠나 — 재현 가치가 없다

- **학습된 가중치가 하나도 남아 있지 않다.** `models/` 는 비어 있고 `*.h5` 는
  `.gitignore` 대상이라 저장소에 들어온 적이 없다. `saved_models/` 는 존재하지 않는다.
- **정량 지표가 하나도 기록되지 않았다.** 5-fold × 10 epoch 학습을 완주한 흔적은
  있었지만, 노트북 출력에 남은 것은 `saving model to ...` 줄뿐이었다. accuracy·AUC·
  precision·recall·F1·혼동행렬 어느 것도 없다. → **성능을 주장할 근거가 전혀 없다.**
- **기록된 예측은 단 2건**이고, 그중 하나는 코골이 오디오에서 잘라낸 세그먼트를
  `No Snoring`(0.135)으로 판정한 것이라 **오탐(false negative)으로 보인다.**
- **그대로는 실행되지 않았다.** `training/model_train.py` 는
  `from data_loader import load_data` 인데 `data_loader.py` 는 `utils/` 에 있었고,
  `inference/inference.py` 는 `from model_train import create_model` 로 다른 폴더를
  가리켰다. 둘 다 ImportError.
- **스택이 어긋났다.** `requirements.txt` 는 TensorFlow/Keras 2.10 대역으로 고정인데
  실제 실행 로그는 Keras 3 경로였다. 후속 프로젝트(snore_dsp)는 uv + Python 3.11 로
  환경 자체가 다르다.

즉 "고쳐서 이어 쓰는" 것이 아니라 **새로 쓰는 게 맞는 코드**였다.

---

## 2. 참고용 — 무엇을 했었나

| 항목 | 내용 |
|---|---|
| 과제 | 코골이/비코골이 **이진 분류**, 멜스펙트로그램 **이미지 분류**로 환원 |
| 입력 | 224×224×3 (dB 멜을 0~255 uint8 로 정규화 후 3채널 복제) |
| 구조 | `Conv2D(32)-BN-MaxPool` → `Conv2D(64)-BN-MaxPool` → `Conv2D(128)-BN-MaxPool` → `Flatten` → `Dense(64, relu)` → `Dense(1, sigmoid)` |
| 손실/최적화 | `binary_crossentropy`, `adam`, batch 32, 10 epoch |
| 검증 | `KFold(n_splits=5, shuffle=True, random_state=82)` |
| 라벨 | 폴더 기반 (`<dataset>/1` = 코골이, `<dataset>/0` = 비코골이) |
| 데이터 | 오픈소스 **1초** 클립 1000개 (코골이 500 / 비코골이 500) |
| 추적 | wandb (`project="Easy-Sleep"`) |

전처리 레시피 자체는 삭제하지 않았다 → [`utils/clip_features.py`](../utils/clip_features.py)
(`clip_to_logmel` + `logmel_to_image`)에 단일 소스로 남아 있고, 학습·추론이 이것을
공유하도록 설계돼 있다.

추론 쪽 설계도 기록해 둔다: 녹음 전체를 `CLIP_SEC` 단위로 윈도잉해 창마다 확률을 뽑고,
**창별 확률의 최댓값(peak)으로 파일 단위 판정**을 했다(평균이 아님). 재구현 시 이
집계 방식은 재검토 대상이다 — peak 는 민감하지만 단발 오탐에 취약하다.

---

## 3. 재학습 때 반복하지 말아야 할 결함

발견해 놓고 고치지 않은 것들이다. 새 코드에서는 처음부터 피할 것.

1. **KFold 가 stratified 가 아니었고, 녹음 단위 그룹핑이 없었다.**
   클립 단위로 무작위 분할하면 **같은 녹음(같은 사람)에서 나온 클립이 train 과 val
   양쪽에 들어간다.** 성능이 부풀려진다. → `StratifiedGroupKFold` 로 **사람/녹음을
   그룹 키로** 묶을 것.
2. **`ModelCheckpoint` 가 `save_best_only=False` 인데 파일명은 `best_model_*`.**
   저장된 건 best 가 아니라 **마지막 epoch** 이었다. 이름과 동작이 어긋나 있었다.
3. **`specificity()` · `f1_score()` 를 구현만 하고 `metrics=[...]` 에 넣지 않았다.**
   지표가 하나도 남지 않은 원인 중 하나. 민감도/특이도는 이 과제의 핵심 지표이므로
   반드시 연결하고, **에폭별 로그를 파일로 남길 것**(노트북 출력에만 의존 금지).
4. **`k_fold_splits.py` 에 `images[train_idx]` 를 두 번 넘겨 라벨 자리에 이미지가
   들어가는 버그**가 있었다(뒤늦게 수정). 분할 함수는 shape 를 assert 할 것.
5. **경로 하드코딩** — `/mnt/d/Snoring/...`(WSL) 과 `D:\Snoring\...`(Windows) 가
   같은 저장소에 공존했다. wandb run id 도 하드코딩(`id='spdszfcu'`).

---

## 4. 재학습 전에 **먼저 결정해야 할 것**

### 4-1. 클립 길이: 1초냐 4초냐 (미해결)

정면으로 충돌하는 두 근거가 있다.

- `clip_features.py` 의 결론: **`CLIP_SEC = 4.0`**. 코골이 판별의 결정적 단서는
  "드르렁–쉼–드르렁" **호흡 주기(2~6초)** 이고, 4초 창이어야 CNN 이 그 리듬을
  학습할 수 있다. 1초는 오픈소스 데이터셋이 1초였던 것뿐이다.
  규칙기반 검출기가 **주기성**을 핵심 신호로 쓰는 것과도 일관된다.
- 현실: **라벨이 붙은 데이터는 그 1초 클립뿐**이다. 4초로 가려면 자체 데이터에
  구간 라벨이 필요하다.

→ 1초로 학습하면 모델은 코골이의 **음색**만 배우고 **반복 패턴**은 못 배운다.
이 트레이드오프를 먼저 정하지 않으면 코드를 어떻게 짜도 다시 엎게 된다.

> **주의**: 학습과 추론이 다른 `CLIP_SEC` 을 쓰면 "픽셀당 시간"이 달라져 skew 가
> 난다. `hop_sec` 은 달라도 무방하다.

### 4-2. 오픈소스 데이터셋의 채널 편향 (leakage 위험)

`stat_snore/opensource_data/Snoring_Dataset_16000hz` 실측:

| 클래스 | 모노 | 스테레오 | 길이 |
|---|---|---|---|
| snoring (500) | 122 | 378 (76%) | 전부 1.0초 |
| no_snoring (500) | 330 | 170 (34%) | 전부 1.0초 |

**채널 수만으로 약 71% 정확도가 나온다.** 로딩 시 mono 변환을 명시적으로 고정하지
않으면 모델이 포맷 아티팩트로 치팅할 수 있다. `librosa.load(..., mono=True)` 를
확실히 쓰고, 전처리 후 채널 정보가 남지 않는지 확인할 것.

---

## 5. 삭제한 파일

| 파일 | 성격 |
|---|---|
| `training/model_train.py` | CNN 정의 + 학습 루프 + wandb |
| `training/k_fold_splits.py` | 5-fold 분할 |
| `utils/data_loader.py` | `image_dataset_from_directory` 로더 |
| `inference/inference.py` | 창별 예측 + peak 집계 |
| `notebooks/train_test2.ipynb` | 학습 실행 노트북 (환자 오디오 임베드 포함) |

복구 가능성은 파일마다 다르다.

- `training/model_train.py`, `training/k_fold_splits.py`, `utils/data_loader.py`
  → git 이력에 커밋된 상태로 남아 있어 되살릴 수 있다.
- `inference/inference.py` → **이력에는 옛 버전(존재하지 않는 모듈
  `feature_extract_save_img` 를 import 하던 깨진 상태)만 있다.** `clip_features` 를
  공유하도록 고친 버전은 커밋되기 전에 삭제됐으므로 이력에서 되살릴 수 없다.
  그 설계(창별 예측 → peak 집계, 학습과 동일한 `clip_to_logmel`/`logmel_to_image`
  사용)는 위 2장에 옮겨 두었다.
- `notebooks/train_test2.ipynb` → **개인정보 정리 과정에서 이력에서도 제거**했으므로
  되살릴 수 없다. 그 노트북에 있던 내용은 이 문서 2·3장에 옮겨 두었다.
