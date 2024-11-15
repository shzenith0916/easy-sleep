import os
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
from librosa import display
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
from model_train import create_model  # call the create_model method
from feature_extract_save_img import load_audio_and_resample, remove_dc_offset, rms_normalize_audio, reduce_noise, create_mel_spectrogram, mel_spec_to_db
import warnings

# TensorFlow 경고 수준 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: 모든 로그, 1: 정보 로그 제거, 2: 경고 제거, 3: 오류만 표시
warnings.filterwarnings("ignore", category=FutureWarning) 

# GPU 확인 및 세팅
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 첫 번째 GPU만 설정
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")



# 2. 오디오 데이터 로드 및 멜 스펙트로그램 생성
SAMPLE_RATE = 16000 # sample_rate (int): Desired sample rate for resampling.
N_FFT = 2048 # n_fft (int): Fourier Transform window size. window 간 이동 간격
HOP_LENGTH = 256 # hop_length (int): Step size between windows.
N_MELS = 128 # n_mels (int): Number of Mel bands. mel-spectrogram의 주파수 대역 수

def process_audio_to_mel_image(file_path, sample_rate=SAMPLE_RATE, n_fft = N_FFT, hop_length = HOP_LENGTH, n_mels = N_MELS):
        """
        Processes an audio file to generate a dB-scaled mel-spectrogram.

        Parameters:
            file_path (str): Path to the audio file.
        
        Returns:
            db_scaled_mel (np.ndarray): dB-scaled mel-spectrogram.
            sample_rate (int): Sampling rate of the processed audio.
        """
        
        audio, sr = load_audio_and_resample(file_path, sample_rate)
        dc_offset_removed = remove_dc_offset(audio)
        rms_normalized_data = rms_normalize_audio(dc_offset_removed)
        denoised_data = reduce_noise(rms_normalized_data, sr)
        mel_spectrogram = create_mel_spectrogram(denoised_data, sr, n_fft, hop_length, n_mels)
        db_scaled_mel = mel_spec_to_db(mel_spectrogram)

        return db_scaled_mel, sr, hop_length


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

def make_prediction(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction


# inference.py 테스트용 오디오 파일 경로
file_path = os.path.join(os.getcwd(), 'output_segment_1.wav')
file_name = Path(file_path).stem
test_image_path = f"{file_name}.png"

# 파일 확인
if not os.path.exists(file_path):
    raise FileNotFoundError("Audio file not fond at {}".format(file_path))

# 모델 생성 및 로드
model = create_model() 
try:
    model.load_weights("./saved_models/best_model_fold_2.weights.h5")
except Exception as e:
    raise RuntimeError(f"Error loading model weights: {e}")

# Mel-spectrogram 생성 및 시각화
db_scaled_mel, sample_rate, hop_length = process_audio_to_mel_image(file_path)
plt.figure(figsize=(10, 4))
display.specshow(db_scaled_mel, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("{} Mel-spectrogram".format(file_name))
plt.tight_layout()
plt.savefig(test_image_path)


# 예측 수행 및 출력
prediction_score = make_prediction(test_image_path)
class_label = "Snoring" if prediction_score[0][0] > 0.5 else "No Snoring" # 예를 들어, 0.5 임계값을 기준으로 분류
print("Prediction:", prediction_score)
print("Class: {}".format(class_label))

# 임시 파일 삭제
os.remove(test_image_path)