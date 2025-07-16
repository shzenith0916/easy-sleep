import os
import matplotlib.pyplot as plt
import librosa.display


def visualize_waveform(file_path):
    '''오디오 파일 로드하고, librosa 라이브러리를 사용하여, 오디오 파형과 스펙트로그램을 시각화.
        sr 변수는 sampling rate를 의미
        sr=None으로 설정: 오디오가 원래 샘플링 레이트로 로드되어서, 오디오 원본 특성 유지 가능. 
        librosa.display.waveshow를 사용하여 오디오의 파형을 시각화하여, 시간에 따른 소리의 크기와 강도를 이해하는데 도움.
        y축에 amplitude 진폭과 x축에 시간을 사용. 
    '''
    # 오디오 로드
    audio, sampling_rate = librosa.load(
        file_path, sr=None)  # 파일의 원래 샘플링 레이트 출력하려면, None 넣기

    # 파형 시각화
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sampling_rate)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), 'output.wav')
    visualize_waveform(file_path)
