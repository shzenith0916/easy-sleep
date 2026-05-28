# EDA 정리 — 코골이 검출 데이터 탐색 기록

`first_eda` / `second_eda` / `third_eda` 세 탐색 노트북과, 모델 입력 전처리(파이프라인 B) 프로토타입인 `feature_extract_test` 가 무엇을 했고, 무엇을 결론냈으며, 어떤 버그·주의점이 있었는지 한 곳에 정리한다.

> **이 노트북들은 "데이터 탐색용 노트북"이다.** 일부 단순 버그·경로는 정리됐지만(아래 각 절의 "수정됨" 참고) 완전한 재실행이 목적은 아니다 — `first_eda`/`third_eda`는 여전히 옛 절대경로(`Documents\코골이\...`, 없는 파일 `PSG_sample...wav`)를 가리키고, `second_eda`는 경로를 `data/raw`로 고쳤지만 8시간대 파일용 파라미터(예: 노이즈 프로파일 140~150분 구간)가 짧은 샘플과 맞지 않는다. 그래서 결론과 **남아 있는** 주의점을 여기에 박제한다.
>
> 검출 로직은 이미 규칙기반 [../preprocessing/snore_extract.py](../preprocessing/snore_extract.py) 로 대체됐다. 전체 데이터 흐름은 [pipeline_overview.ipynb](pipeline_overview.ipynb) 참고.

---

## EDA1 — [first_eda.ipynb](first_eda.ipynb) : 멜스펙트로그램 + dB 임계값 검출 (초기 시도)

**한 일**
- `output.wav` 로드(44.1kHz) → 멜스펙트로그램 → 200~1100Hz 대역 평균 dB로 임계값 검출
- 임계값 초과 연속 프레임을 묶어 클립으로 저장

**배운 것 / 결론**
- 절대 dB 임계값(평균+여유)만으로는 단발 소음에 쉽게 속는다는 한계를 확인.
- 이 한계가 이후 **적응형 에너지 + 저주파 비중 + 호흡 주기성(AND)** 방식([../preprocessing/snore_extract.py](../preprocessing/snore_extract.py))으로 발전한 출발점.

**⚠️ 남아 있는 버그 (검출 로직 자체가 snore_extract.py로 대체됨 → 수정 안 함)**
- `extract_snoring_segments`: `duration_sec = end - start` 가 `frames_to_samples` 결과(=**샘플 수**)인데 초로 출력·비교한다. 출력의 `29184.00초`는 실제 ≈ 0.66초(29184/44100). → **샘플/초 단위 혼동**.
- 같은 노트북의 다른 분할 셀: `end - start + 1 > min_segment_length`(=1.0초) — **프레임 수와 초**를 비교. 단위 혼동의 또 다른 형태.
- 로드 경로가 옛 절대경로(`...\코골이\...\output.wav`)라 그대로는 실행되지 않음.
- **핵심 교훈**: 오디오에서 *샘플 / 초 / 프레임* 단위를 항상 명시적으로 구분할 것.

> 수정됨: `melspectrogram(..., sr=s)` 의 `s` 오타 → `sr` 로 정정.

---

## EDA2 — [second_eda.ipynb](second_eda.ipynb) : 디노이즈 + 주파수 대역 분석

**한 일**
- `noisereduce` 의 stationary / non-stationary, 노이즈 프로파일 지정 방식 실험
- 주파수 대역별 에너지 분석, "가장 조용한 청크"를 노이즈 프로파일로 쓰는 방법 탐색
- 로그/dB 스케일(진폭 20·log, 전력 10·log) 직관 정리

**배운 것 / 결론**
- 코골이 에너지는 **저~중주파(20~2000Hz)** 에 몰리고 2000Hz↑는 거의 비어 있음 → 고주파(말소리 등) 제거 여지 확인.
- 긴 파일의 배경 소음은 대체로 정상(stationary)에 가깝다는 관찰.

**⚠️ 남아 있는 버그**
- `spectrogram_comparison(sleep_audio, noise_reduced1, ...)` 호출의 `sleep_audio`, `noise_reduced1` 가 정의된 적 없음 → 그대로 실행하면 `NameError`.
- 8시간대 파일 기준 파라미터(노이즈 프로파일 140~150분 등)가 남아 있어, `data/raw`의 짧은 샘플에는 구간이 맞지 않음.

---

## EDA3 — [third_eda.ipynb](third_eda.ipynb) : 디노이즈 방법 비교

**한 일**
- stationary / non-stationary, DC offset 제거 + RMS 정규화, Wiener, median, Kalman, bandpass 를 파형·스펙트로그램·청취로 비교

**배운 것 / 결론**
- **stationary 디노이즈가 가장 무난**, bandpass(저주파 통과)도 유효.
- Wiener / Kalman 은 코골이 오디오에 부적합.
- RMS 정규화는 노이즈를 증폭할 수 있고, DC offset 제거는 저주파(호흡) 성분을 손상시킬 수 있음 → 신중히.

**⚠️ 남아 있는 버그** 
- `cell-12` `stationary_denoise`: `processed_segments.append(...)` 가 for 루프 **밖**에 있어 마지막 구간만 저장된다(결과 길이가 왜곡됨).
- `cell-49/50` `find_quiet_noise_profile`: `chunk_data` 에 아무것도 담지 않은 채 `min(chunk_data, ...)` 호출 → 빈 시퀀스 에러.

---

## 특징 추출 — [feature_extract_test.ipynb](feature_extract_test.ipynb) : 1초 클립 → 멜/MFCC (파이프라인 B 프로토타입)

> 이건 EDA(탐색)가 아니라 **모델 입력 전처리(B) 프로토타입**이다. 여기서 시도한 레시피가 이후 `utils/clip_features.py`(`clip_to_logmel` + `logmel_to_image`)로 정리돼 학습·추론이 공유한다. (A=라벨 생성 / B=모델 입력 구분은 [pipeline_overview.ipynb](pipeline_overview.ipynb) 참고)

**한 일 (전처리 레시피)**
- 16kHz 로드 → `remove_dc_offset` → `rms_normalize_audio`(target_rms=0.1) → `reduce_noise` → 멜스펙트로그램(n_fft=2048, hop=256, n_mels=128) → dB → PNG 저장
- 짧은 클립용 대안으로 스펙트로그램에 `scipy.ndimage.median_filter(size=(1,3))` 적용
- MFCC(`n_mfcc=20`) 추출도 시험

**배운 것 / 결론**
- 긴 파일(병원 데이터 등, >1초)은 `reduce_noise`(spectral gating) 가능하지만, 이미 1초로 잘린 오픈소스 클립은 노이즈 프로파일을 따로 뗄 수 없어 spectral gating 부적합 → median filter 로 대체 가능하다고 정리.

**⚠️ 주의 (버그라기보다 skew/설계 이슈)**
- `reduce_noise(data, sr)` 가 노이즈 프로파일로 `data[:sr]`(앞 1초)를 쓴다. 테스트 클립이 1초 이하라 **클립 전체가 자기 자신의 노이즈 프로파일**이 되는 degenerate 상태 → 1초 클립에선 의미가 약하다.
- 같은 함수가 인자 `data` 를 받고도 내부에선 전역 `audio_data` 를 denoise한다 → 넘긴 정규화 신호가 아니라 원본을 처리하는 모순.
- **핵심**: 디노이즈 방식을 하나로 못박아 학습==추론에서 동일하게 쓸 것 → **해결됨**: 현재는 `clip_to_logmel`(DC offset만, RMS는 no-op·reduce_noise 제외)로 단일화. 과거 `inference.py`가 import하던 누락 모듈 `feature_extract_save_img` 도 clip_features로 대체해 해소.

---

## 종합 — 지금 파이프라인에 반영된 것

1. **핵심 대역**: 코골이는 저~중주파. `snore_extract.py` 의 `SNORE_BAND=(80, 600)` 이 EDA의 관찰(200~1100 / 20~2000Hz)과 정합.
2. **디노이즈는 "방법 선택"보다 "학습==추론에서 동일하게 고정"이 더 중요.** → 이제 `utils/clip_features.py`의 `clip_to_logmel`(DC offset만, RMS는 no-op·reduce_noise 제외)로 단일화돼 학습·추론이 공유한다. (과거엔 inference/snore_extract/EDA가 디노이즈 3종 혼재였음.) 흐름은 [pipeline_overview.ipynb](pipeline_overview.ipynb) 참고.
3. **단위 구분**(샘플/초/프레임)은 EDA1 버그가 남긴 교훈.
