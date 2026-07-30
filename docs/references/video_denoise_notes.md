파이썬으로 영상에서 오디오 잡음을 제거하는 몇 가지 방법을 소개해드리겠습니다.

## 1. librosa와 noisereduce를 사용한 기본적인 잡음 제거

```python
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

def remove_noise_from_video(video_path, output_path, noise_reduction_strength=0.8):
    """
    영상에서 오디오를 추출하고 잡음을 제거한 후 다시 합성
    """
    # 영상에서 오디오 추출
    video = VideoFileClip(video_path)
    audio = video.audio
    
    # 임시 오디오 파일로 저장
    temp_audio = "temp_audio.wav"
    audio.write_audiofile(temp_audio, verbose=False, logger=None)
    
    # 오디오 로드
    y, sr = librosa.load(temp_audio, sr=None)
    
    # 잡음 제거 (첫 1초를 잡음 샘플로 사용)
    noise_sample = y[:sr]  # 첫 1초
    reduced_noise = nr.reduce_noise(
        y=y, 
        sr=sr, 
        y_noise=noise_sample,
        prop_decrease=noise_reduction_strength
    )
    
    # 처리된 오디오 저장
    cleaned_audio = "cleaned_audio.wav"
    sf.write(cleaned_audio, reduced_noise, sr)
    
    # 새 오디오와 원본 비디오 합성
    new_audio = AudioFileClip(cleaned_audio)
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_path, verbose=False, logger=None)
    
    # 정리
    video.close()
    audio.close()
    new_audio.close()
    final_video.close()
    
    import os
    os.remove(temp_audio)
    os.remove(cleaned_audio)

# 사용 예시
remove_noise_from_video("input_video.mp4", "output_video.mp4")
```

## 2. 고급 잡음 제거 - Spectral Subtraction

```python
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import wiener
from moviepy.editor import VideoFileClip, AudioFileClip

def spectral_subtraction_denoising(audio_path, output_path, alpha=2.0, beta=0.01):
    """
    스펙트럼 감산법을 사용한 잡음 제거
    """
    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=None)
    
    # STFT 변환
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # 첫 0.5초를 잡음 추정으로 사용
    noise_frames = int(0.5 * sr / 512)
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # 스펙트럼 감산
    subtracted_magnitude = magnitude - alpha * noise_spectrum
    
    # 음수 값 방지
    subtracted_magnitude = np.maximum(subtracted_magnitude, 
                                    beta * magnitude)
    
    # 위상 정보와 결합
    enhanced_stft = subtracted_magnitude * np.exp(1j * phase)
    
    # 역 STFT
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
    
    # 저장
    sf.write(output_path, enhanced_audio, sr)
    
    return enhanced_audio, sr

def advanced_video_denoising(video_path, output_path):
    """
    고급 잡음 제거를 적용한 영상 처리
    """
    # 영상에서 오디오 추출
    video = VideoFileClip(video_path)
    audio = video.audio
    
    temp_audio = "temp_audio.wav"
    audio.write_audiofile(temp_audio, verbose=False, logger=None)
    
    # 스펙트럼 감산법 적용
    enhanced_audio, sr = spectral_subtraction_denoising(temp_audio, "enhanced_audio.wav")
    
    # 추가적인 Wiener 필터 적용
    filtered_audio = wiener(enhanced_audio, noise=0.1)
    
    # 최종 오디오 저장
    final_audio_path = "final_audio.wav"
    sf.write(final_audio_path, filtered_audio, sr)
    
    # 영상과 합성
    new_audio = AudioFileClip(final_audio_path)
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_path, verbose=False, logger=None)
    
    # 정리
    video.close()
    audio.close()
    new_audio.close()
    final_video.close()
    
    import os
    os.remove(temp_audio)
    os.remove("enhanced_audio.wav")
    os.remove(final_audio_path)

# 사용 예시
advanced_video_denoising("input_video.mp4", "output_video.mp4")
```

## 3. 실시간 처리 및 다양한 필터 조합

```python
import librosa
import noisereduce as nr
import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip

class AudioDenoiser:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        
    def bandpass_filter(self, audio, lowcut=80, highcut=8000):
        """
        대역통과 필터 (음성 주파수 범위만 통과)
        """
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, audio)
    
    def adaptive_noise_reduction(self, audio, noise_gate_threshold=0.01):
        """
        적응형 잡음 제거
        """
        # 노이즈 게이트 적용
        audio_gated = np.where(np.abs(audio) < noise_gate_threshold, 0, audio)
        
        # 통계적 잡음 제거
        reduced = nr.reduce_noise(
            y=audio_gated,
            sr=self.sr,
            stationary=False,  # 비정적 잡음 처리
            prop_decrease=0.8
        )
        
        return reduced
    
    def enhance_speech(self, audio):
        """
        음성 개선을 위한 추가 처리
        """
        # 대역통과 필터 적용
        filtered = self.bandpass_filter(audio)
        
        # 적응형 잡음 제거
        denoised = self.adaptive_noise_reduction(filtered)
        
        # 음량 정규화
        if np.max(np.abs(denoised)) > 0:
            denoised = denoised / np.max(np.abs(denoised)) * 0.8
        
        return denoised
    
    def process_video(self, video_path, output_path):
        """
        영상 전체 처리
        """
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # 임시 오디오 추출
        temp_audio = "temp_audio.wav"
        audio.write_audiofile(temp_audio, verbose=False, logger=None)
        
        # 오디오 로드 및 처리
        y, sr = librosa.load(temp_audio, sr=self.sr)
        enhanced_audio = self.enhance_speech(y)
        
        # 처리된 오디오 저장
        processed_audio = "processed_audio.wav"
        sf.write(processed_audio, enhanced_audio, sr)
        
        # 영상과 합성
        new_audio = AudioFileClip(processed_audio)
        final_video = video.set_audio(new_audio)
        final_video.write_videofile(output_path, verbose=False, logger=None)
        
        # 정리
        video.close()
        audio.close()
        new_audio.close()
        final_video.close()
        
        import os
        os.remove(temp_audio)
        os.remove(processed_audio)

# 사용 예시
denoiser = AudioDenoiser(sample_rate=22050)
denoiser.process_video("input_video.mp4", "output_video.mp4")
```

## 필요한 라이브러리 설치

```bash
pip install librosa noisereduce soundfile moviepy scipy numpy
```

## 사용 팁

1. **잡음 샘플 선택**: 첫 몇 초의 무음 구간을 잡음 샘플로 사용하면 더 좋은 결과를 얻을 수 있습니다.

2. **매개변수 조정**: 
   - `prop_decrease`: 잡음 제거 강도 (0.5-0.9)
   - `alpha`: 스펙트럼 감산 강도 (1.5-3.0)
   - `noise_gate_threshold`: 노이즈 게이트 임계값

3. **성능 최적화**: 긴 영상의 경우 청크 단위로 처리하여 메모리 사용량을 줄일 수 있습니다.

어떤 종류의 잡음(배경 소음, 바람 소리, 전자 잡음 등)인지 알려주시면 더 특화된 솔루션을 제안해드릴 수 있습니다!