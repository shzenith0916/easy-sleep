# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# 폴더구조: /snoring_data_process/0 /snoring_data_process/1  0 -> no snoring 1-> snoring
dataset_path = r"D:\Snoring\easy_sleep\snoring_data_process"


def load_data(dataset_path, img_size=(224, 224), batch_size=32):  # (224,224), (480,480), (640,640)
    # 데이터셋 생성
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        labels="inferred",               # 폴더 이름을 기반으로 자동 라벨링
        label_mode="int",                # 라벨을 정수형으로 설정
        batch_size=batch_size,
        image_size=img_size,           # 이미지 크기를 모델 입력 크기에 맞게 조정
        shuffle=True
    )

    # Numpy 배열로 추출
    images = []
    labels = []

    # 데이터 확인
    for img_batch, label_batch in dataset:
        images.extend(img_batch.numpy())  # append를 사용하면 각 배치가 서브 리스트 형태로 추가
        labels.extend(label_batch.numpy())

    # Numpy 배열로 변환 후 shape 출력
    images_array = np.array(images)
    labels_array = np.array(labels)

    print("Image batch shape:", images_array.shape)
    print("Label batch shape:", labels_array.shape)

    return images_array, labels_array
