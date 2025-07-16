# -*- coding: utf-8 -*-
import os
import warnings
import tensorflow as tf
from keras import backend as K
from data_loader import load_data
from k_fold_splits import kfold_splits
import wandb
from wandb.integration.keras import WandbCallback
import logging 


# Config 세팅
CONFIG = {"project_name": "Easy-Sleep",
    "dataset_path": "/mnt/d/Snoring/easy_sleep/snoring_data_process",
    "epochs": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "n_folds": 5,
    "save_model_path": "./saved_models"
}


def create_model(input=(224, 224, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# Custom 지표
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



def train_model(model, train_dataset, val_dataset, config, fold):

    # 모델 컴파일
    model.compile(
        optimizer=config['optimizer'],
        loss='binary_crossentropy',
        metrics=[
            'accuracy',                    # 내장 accuracy 메트릭
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),       # recall=sensitivity 내장 메트릭
            tf.keras.metrics.AUC(name='auc')                          # 내장 AUC 메트릭
        ]
    )

    # ModelCheckpoint 콜백 정의 - fold 번호를 파일 경로에 포함
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
        filepath="{}/best_model_fold_{}.weights.h5".format(config['save_model_path'], fold),   # fold별 다른 파일 이름
        monitor='val_accuracy',
        mode='max',
        save_best_only=False,
        verbose=1,
        save_weights_only=True),
        WandbCallback()
        ]
    
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=callbacks,  
        verbose=1  
    )

    # Evaluate the model after training
    result = model.evaluate(val_dataset)
    metrics = dict(zip(model.metrics_names, result))
    wandb.log(metrics)

    return history, result


if __name__ == "__main__":
    
    # Initialize wandb at the beginning of your script
    # wandb.init(project="Easy-Sleep", config={
    #     "epochs": 10,
    #     "batch_size": 32,
    #     "optimizer": "adam",
    #     # Add other hyperparameters here
    # })

    wandb.init(project="Easy-Sleep", name = 'test_train', id='spdszfcu', resume='allow')

    # TensorFlow 경고 수준 설정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: 모든 로그, 1: 정보 로그 제거, 2: 경고 제거, 3: 오류만 표시
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    warnings.filterwarnings("ignore", category=FutureWarning) 
    logging.getLogger('tensorflow').setLevel(logging.ERROR)


    # 데이터 로드
    images, labels = load_data(CONFIG['dataset_path'])

    # K-Fold 분할 및 학습
    folds = kfold_splits(images, labels, n_folds=CONFIG['n_folds']) # folds value in (train_dataset, val_dataset)
    
    for fold, (train_dataset, val_dataset) in enumerate(folds, start=1):
        print("Training Fold {}/{}".format(fold, CONFIG['n_folds']))

        # 모델 생성 및 학습
        model = create_model()
        history, result = train_model(model, train_dataset, val_dataset, config=CONFIG, fold=fold) 

    # Finish the WandB run
    wandb.finish()