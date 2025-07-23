# FFmpeg 사용 오디오 추출 가이드

## 🎬➡️🎵 FFmpeg로 비디오를 오디오로 변환

### 기본 변환 명령어

#### WAV 형식 (무손실)
```bash
# 기본 변환
ffmpeg -i video_clip0001.avi video_clip0001.wav

# 고품질 변환 (44.1kHz, 16bit, 스테레오)
ffmpeg -i video_clip0001.avi -ar 44100 -ac 2 -sample_fmt s16 video_clip0001.wav

# 코골이 분석용 (16kHz, 모노)
ffmpeg -i video_clip0001.avi -ar 16000 -ac 1 -sample_fmt s16 video_clip0001.wav
```

#### MP3 형식 (압축)
```bash
# 기본 MP3 변환
ffmpeg -i video_clip0001.avi -acodec mp3 -ab 192k video_clip0001.mp3

# 고품질 MP3 (16-bit 처리 후 192k 압축)
ffmpeg -i video_clip0001.avi -sample_fmt s16 -acodec mp3 -ab 192k video_clip0001.mp3
```

#### 기타 형식
```bash
# 무손실 FLAC
ffmpeg -i video_clip0001.avi -acodec flac video_clip0001.flac

# AAC 형식
ffmpeg -i video_clip0001.avi -sample_fmt s16 -acodec aac -ab 128k video_clip0001.aac
```

## 📊 주요 옵션 설명

### Sample Format (`-sample_fmt`)

| 포맷  | 의미                  | 범위                           | 용도                 |
| ----- | --------------------- | ------------------------------ | -------------------- |
| `s16` | 16-bit signed integer | -32,768 ~ 32,767               | **일반적인 CD 품질** |
| `s24` | 24-bit signed integer | -8,388,608 ~ 8,388,607         | 고품질 녹음          |
| `s32` | 32-bit signed integer | -2,147,483,648 ~ 2,147,483,647 | 전문가용             |
| `flt` | 32-bit float          | -1.0 ~ 1.0                     | 오디오 처리용        |
| `dbl` | 64-bit double         | -1.0 ~ 1.0                     | 최고 정밀도          |

**s16 = Signed 16-bit Integer**
- **s**: Signed (부호 있는 정수)
- **16**: 16비트
- **CD 품질과 동일**, 적당한 파일 크기, 높은 호환성

### Audio Bitrate (`-ab`)

| 비트레이트 | 품질      | 용도           | 파일 크기  |
| ---------- | --------- | -------------- | ---------- |
| `64k`      | 낮음      | 음성, 라디오   | 매우 작음  |
| `128k`     | 보통      | 일반적인 MP3   | 작음       |
| `192k`     | **좋음**  | **고품질 MP3** | **적당함** |
| `256k`     | 매우 좋음 | 고급 MP3       | 큼         |
| `320k`     | 최고      | 최고 품질 MP3  | 가장 큼    |

**192k = 192 kbps (kilobits per second)**
- 1초당 192,000비트의 데이터로 오디오를 인코딩
- 품질과 파일 크기의 좋은 균형점

### 기타 옵션

- **`-i`**: 입력 파일
- **`-ar 44100`**: 샘플링 레이트 44.1kHz
- **`-ac 2`**: 오디오 채널 수 (2 = 스테레오, 1 = 모노)
- **`-acodec`**: 오디오 코덱 지정

## 🔧 두 옵션 함께 사용하기

### ✅ **MP3/AAC 등 압축 포맷** (둘 다 의미 있음)
```bash
# 16-bit 샘플로 처리 후 192k로 압축
ffmpeg -i video_clip0001.avi -sample_fmt s16 -acodec mp3 -ab 192k output.mp3

# 32-bit float로 고정밀 처리 후 256k로 압축  
ffmpeg -i video_clip0001.avi -sample_fmt flt -acodec mp3 -ab 256k output.mp3
```

### ⚠️ **WAV 등 무손실 포맷** (sample_fmt만 의미 있음)
```bash
# WAV는 무손실이므로 -ab는 무시됨
ffmpeg -i video_clip0001.avi -sample_fmt s16 -ab 192k output.wav
# 실제로는 -sample_fmt s16만 적용됨

# 올바른 방법
ffmpeg -i video_clip0001.avi -sample_fmt s16 output.wav
```

## 📋 포맷별 적절한 사용법

| 출력 포맷 | sample_fmt | ab       | 예시                       |
| --------- | ---------- | -------- | -------------------------- |
| **WAV**   | ✅ 사용     | ❌ 무시됨 | `-sample_fmt s16`          |
| **MP3**   | ✅ 사용     | ✅ 사용   | `-sample_fmt s16 -ab 192k` |
| **FLAC**  | ✅ 사용     | ❌ 무시됨 | `-sample_fmt s16`          |
| **AAC**   | ✅ 사용     | ✅ 사용   | `-sample_fmt s16 -ab 128k` |

## 💡 코골이 프로젝트 추천 설정

```bash
# WAV로 변환 (무손실, 분석용) - 추천
ffmpeg -i video_clip0001.avi -ar 16000 -ac 1 -sample_fmt s16 video_clip0001.wav

# MP3로 변환 (저장 공간 절약용)
ffmpeg -i video_clip0001.avi -ar 16000 -ac 1 -sample_fmt s16 -acodec mp3 -ab 128k video_clip0001.mp3

# 기본 설정으로 간단하게
ffmpeg -i video_clip0001.avi video_clip0001.wav
```

## 📈 파일 크기 예시 (5분 음악 기준)
- **64k MP3**: 약 2.4MB
- **128k MP3**: 약 4.8MB  
- **192k MP3**: 약 7.2MB
- **320k MP3**: 약 12MB
- **WAV (16-bit/44.1kHz)**: 약 50MB

## 🎯 결론

- **WAV**: 무손실 포맷, 분석용으로 최적
- **MP3**: 압축 포맷, 저장 공간 절약
- **sample_fmt s16**: CD 품질, 대부분의 용도에 적합
- **ab 192k**: 고품질 MP3의 표준
- **압축 포맷에서만 sample_fmt와 ab를 함께 사용하는 것이 의미 있음** 