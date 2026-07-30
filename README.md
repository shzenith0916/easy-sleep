# easy_sleep — 코골이 검출 선행 연구

수면 녹음에서 코골이 구간을 찾아내는 **신호처리·전처리 연구 기록**이다.
후속 프로젝트(`../stat_snore/snore_dsp`)의 출발점이 된 저장소이며, 지금은
**참조용 아카이브**로 유지한다.

> ⚠️ **학습된 모델도, 성능 지표도 없다.** 2024-11 의 CNN 학습 시도는 가중치·지표가
> 남지 않아 폐기했다. 경위와 재학습 시 주의점은
> [notebooks/model_training_lessons.md](notebooks/model_training_lessons.md) 에 정리했다.
> 이 저장소의 가치는 모델이 아니라 **전처리 설계와 그 근거**에 있다.

---

## 무엇을 알아냈나 — 핵심 4가지

1. **"전처리"는 두 층이다.** A(라벨 생성)와 B(모델 입력)를 분리해야 한다.
   A 는 오프라인 1회, B 는 학습과 추론이 반드시 동일해야 한다.
   → [notebooks/pipeline_overview.ipynb](notebooks/pipeline_overview.ipynb)
2. **코골이 검출의 결정적 단서는 호흡 주기성이다.** 절대 dB 임계값은 문 닫는 소리
   같은 단발 소음에 쉽게 속는다. 적응형 에너지 + 저주파 비중(80\~600Hz) +
   주기성(1.5\~7초)을 **AND** 로 묶어야 한다.
   → [preprocessing/snore_extract_explanation.md](preprocessing/snore_extract_explanation.md)
3. **디노이즈는 "방법 선택"보다 "학습==추론에서 고정"이 중요하다.** stationary 가
   가장 무난했고 Wiener·Kalman 은 부적합. 최종 레시피는 DC offset 제거만 남겼다.
   → [notebooks/EDA.md](notebooks/EDA.md)
4. **클립 길이는 4초여야 한다.** "드르렁–쉼–드르렁" 한 주기가 창 안에 들어와야
   CNN 이 리듬을 배운다. 1초는 오픈소스 데이터셋이 1초였던 것뿐이다.
   → [utils/clip_features.py](utils/clip_features.py)

---

## 재사용할 코드는 네 곳

| 경로 | 역할 |
|---|---|
| [preprocessing/snore_extract.py](preprocessing/snore_extract.py) | **규칙기반 코골이 구간 추출 (A)**. 적응형 3신호 AND. 튜닝값이 파일 상단에 모여 있다. |
| [utils/clip_features.py](utils/clip_features.py) | **모델 입력 전처리 단일 소스 (B)**. `clip_to_logmel` + `logmel_to_image`. 학습·추론이 이걸 공유해야 skew 가 없다. |
| [preprocessing/build_training_images.py](preprocessing/build_training_images.py) | 위 함수로 학습 이미지(`<out>/1`, `<out>/0`) 생성 |
| [preprocessing/modular_noise_analysis.py](preprocessing/modular_noise_analysis.py) | 새 녹음 진단 — 대역 에너지, 노이즈 유형 추정, 제거 강도 권장 |

보조: [preprocessing/denoise.py](preprocessing/denoise.py) (긴 파일용 디노이즈),
[preprocessing/ffmpeg.py](preprocessing/ffmpeg.py) (PSG 영상 → wav),
[preprocessing/rename_videos.py](preprocessing/rename_videos.py) (환자명 익명화),
[utils/audio_convert.py](utils/audio_convert.py) (로더/변환 유틸).

---

## 실행

```bash
uv sync                        # 의존성 설치
uv sync --extra notebooks      # + jupyter (노트북 볼 때)
```

```bash
# 코골이 구간 추출 — 먼저 --dry-run 으로 검출 구간을 눈으로 확인할 것
uv run python preprocessing/snore_extract.py night.m4a --dry-run --csv check.csv
uv run python preprocessing/snore_extract.py ./recordings/*.m4a -o ./clips

# 추출한 클립 → 학습용 멜 이미지
uv run python preprocessing/build_training_images.py \
    --snore ./clips --nosnore ./nosnore_clips -o ./snoring_data_process
```

민감도는 `-s high`(기본, 정밀도 우선) / `medium` / `low`.
m4a·mp3 를 읽으려면 **ffmpeg** 가 PATH 에 있어야 한다.

---

## 데이터는 저장소에 없다

수면 녹음은 환자 개인정보이므로 **버전관리에 포함하지 않는다.**
`data/` 는 전체가 gitignore 대상이며, 필요한 파일은 별도로 받아서 두어야 한다.

```
data/
├── raw/        원본 녹음
└── processed/  잘라낸 클립
```

새 코드를 쓸 때 **파일 경로에 환자 식별정보를 하드코딩하지 말 것.**
PSG 영상 폴더명에 실명이 들어오는 경우 `rename_videos.py` 로 익명화 사본을 만든 뒤 쓴다.

---

## 문서 지도

| 문서 | 내용 |
|---|---|
| [notebooks/EDA.md](notebooks/EDA.md) | 탐색 3회의 결론 + **남아 있는 버그를 일부러 기록해 둔 장부** |
| [preprocessing/snore_extract_explanation.md](preprocessing/snore_extract_explanation.md) | 검출기 설계 근거. "파일 종류로 분기하지 말라"는 논증과 `MERGE_GAP_SEC` 교훈 |
| [notebooks/model_training_lessons.md](notebooks/model_training_lessons.md) | 학습 시도의 실패 경위, 반복하지 말아야 할 결함, 재학습 전 결정할 것 |
| [preprocessing/ffmpeg_audio_extract.md](preprocessing/ffmpeg_audio_extract.md) | 영상 → 오디오 추출 파라미터 (`-ar 16000 -ac 1 -sample_fmt s16`) |
| [preprocessing/videofile_rename.md](preprocessing/videofile_rename.md) | 환자명 익명화 도구 사용법 |

노트북은 **재실행을 보장하지 않는다.** 옛 절대경로와 8시간 파일 기준 파라미터가
남아 있다. 결론은 위 문서에 옮겨 두었으니 그쪽을 먼저 볼 것.

`docs/references/` 는 **이 프로젝트가 작성한 문서가 아니다.** 외부에서 받은 참고
노트(noisereduce 사용법, 영상 잡음 제거 방법)이므로 설계 근거로 인용하지 말 것.
