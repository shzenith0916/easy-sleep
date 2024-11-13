# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


def create_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Input 레이어 추가
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # y_true를 float32로 변환
    y_pred = K.round(y_pred)  # y_pred를 0 또는 1로 반올림 후 float32로 변환
    y_pred = tf.cast(y_pred, tf.float32)

    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # y_true를 float32로 변환
    y_pred = K.round(y_pred)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def train_model(model, train_dataset, val_dataset, epochs=10, optim='adam', callbacks=None):

    model.compile(
        optimizer=optim,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',                    # 내장 accuracy 메트릭
            'precision',
            'recall',                      # recall=sensitivity 내장 메트릭
            specificity,                   # 커스텀 메트릭
            f1_score,                      # 커스텀 메트릭
            'AUC'                          # 내장 AUC 메트릭
        ]
    )

    result = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,  # 외부에서 전달받은 콜백 리스트 사용
        verbose=0  
    )

    return result

