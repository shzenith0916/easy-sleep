# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.model_selection import KFold


def kfold_splits(images, labels, n_folds=5, shuffle=True):

    kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=82)

    fold_data = []

    for train_idx, val_idx in kfold.split(images, labels):
        x_train, x_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # tf.data.Dataset 생성. 파이썬의 반복 가능 객체로 iterator 로 쓸수 있음.
        train_dataset = tf.data.Dataset.from_tensor_slices(  # 여러 tensor로 자름
            (x_train, y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (x_val, y_val)).batch(32)

        fold_data.append((train_dataset, val_dataset))

    return fold_data
