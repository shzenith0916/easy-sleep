`noisereduce` 패키지로 구간별로 다른 강도의 배경 소음을 처리하는 방법들을 알려드리겠습니다.

## 1. 적응형 소음 제거 (Adaptive Noise Reduction)

구간별로 소음 프로파일을 다르게 적용하는 방법입니다:

```python
import librosa
import noisereduce as nr
import numpy as np

def adaptive_noise_reduction(audio_file, sr=22050, chunk_duration=5):
    """
    구간별로 소음 프로파일을 적응적으로 적용
    """
    # 오디오 로드
    y, sr = librosa.load(audio_file, sr=sr)
    
    # 청크 크기 계산
    chunk_size = int(chunk_duration * sr)
    processed_chunks = []
    
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]
        
        # 각 청크에서 소음 구간 자동 감지 (처음 0.5초를 소음으로 가정)
        noise_sample_size = min(int(0.5 * sr), len(chunk) // 4)
        
        # 소음 제거 적용
        reduced_chunk = nr.reduce_noise(
            y=chunk, 
            sr=sr,
            stationary=False,  # 비정상 소음 처리
            prop_decrease=0.8  # 소음 감소 비율
        )
        
        processed_chunks.append(reduced_chunk)
    
    return np.concatenate(processed_chunks), sr
```

## 2. 스펙트럼 분석 기반 동적 처리

주파수 스펙트럼을 분석하여 구간별 소음 특성에 맞게 처리:

```python
def dynamic_noise_reduction(audio_file, sr=22050):
    """
    스펙트럼 분석 기반 동적 소음 제거
    """
    y, sr = librosa.load(audio_file, sr=sr)
    
    # 스펙트로그램 분석
    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    
    # 시간축별 소음 레벨 추정
    noise_floor = np.percentile(magnitude, 20, axis=0)  # 하위 20% 를 소음으로 추정
    
    # 적응형 파라미터 계산
    window_size = sr * 2  # 2초 윈도우
    processed_segments = []
    
    for i in range(0, len(y), window_size):
        segment = y[i:i + window_size]
        
        # 현재 구간의 소음 레벨 추정
        segment_power = np.mean(segment ** 2)
        
        # 소음 레벨에 따른 파라미터 조정
        if segment_power < np.percentile(y**2, 30):  # 조용한 구간
            prop_decrease = 0.9
            stationary = True
        else:  # 시끄러운 구간
            prop_decrease = 0.7
            stationary = False
        
        # 소음 제거 적용
        reduced_segment = nr.reduce_noise(
            y=segment,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        
        processed_segments.append(reduced_segment)
    
    return np.concatenate(processed_segments), sr
```

## 3. 다단계 소음 제거

여러 번의 소음 제거를 통해 점진적으로 처리:

```python
def multi_stage_noise_reduction(audio_file, sr=22050):
    """
    다단계 소음 제거
    """
    y, sr = librosa.load(audio_file, sr=sr)
    
    # 1단계: 전체적인 정상 소음 제거
    y_reduced = nr.reduce_noise(
        y=y, 
        sr=sr,
        stationary=True,
        prop_decrease=0.5
    )
    
    # 2단계: 비정상 소음 제거
    y_reduced = nr.reduce_noise(
        y=y_reduced,
        sr=sr,
        stationary=False,
        prop_decrease=0.7
    )
    
    # 3단계: 구간별 세부 조정
    chunk_size = sr * 3  # 3초 단위
    final_chunks = []
    
    for i in range(0, len(y_reduced), chunk_size):
        chunk = y_reduced[i:i + chunk_size]
        
        # 각 구간의 소음 특성 분석
        chunk_std = np.std(chunk)
        
        if chunk_std > np.std(y_reduced) * 1.5:  # 높은 변동성 구간
            # 더 강한 소음 제거
            chunk = nr.reduce_noise(
                y=chunk,
                sr=sr,
                stationary=False,
                prop_decrease=0.8
            )
        
        final_chunks.append(chunk)
    
    return np.concatenate(final_chunks), sr
```

## 4. 실제 사용 예제

```python
import matplotlib.pyplot as plt
import soundfile as sf

def process_noisy_audio(input_file, output_file):
    """
    전체 처리 파이프라인
    """
    # 방법 선택 (상황에 따라 조합 가능)
    processed_audio, sr = adaptive_noise_reduction(input_file)
    
    # 결과 저장
    sf.write(output_file, processed_audio, sr)
    
    # 시각화 (선택사항)
    plt.figure(figsize=(12, 6))
    
    # 원본 오디오
    original, _ = librosa.load(input_file, sr=sr)
    plt.subplot(2, 1, 1)
    plt.plot(original)
    plt.title('Original Audio')
    
    # 처리된 오디오
    plt.subplot(2, 1, 2)
    plt.plot(processed_audio)
    plt.title('Noise Reduced Audio')
    
    plt.tight_layout()
    plt.show()

# 사용 예제
process_noisy_audio('noisy_audio.wav', 'cleaned_audio.wav')
```

## 추가 팁

1. **소음 샘플 수집**: 각 구간의 시작 부분에서 소음 샘플을 자동으로 추출
2. **파라미터 조정**: `prop_decrease` 값을 0.5-0.9 사이에서 조정
3. **후처리**: 소음 제거 후 음성 향상을 위한 추가 필터링 적용

구간별 소음 강도가 다르다면 `adaptive_noise_reduction` 방법을 먼저 시도해보시고, 필요에 따라 다른 방법들과 조합하여 사용하시면 좋은 결과를 얻을 수 있을 것입니다.