import time
import copy
import tensorflow as tf

import numpy as np
import pickle
import random
from Huristic_AI import HAI
from collections import deque
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential


# 신경망 7개를 하나의 리스트안에 넣고 각각 학습시켜 이용할 수 있도록
# 메모리 - 7개의 신경망 하나달 1개
# 우선순위 메모리 ?

# 대전 테트리스에서 - 상대 를 어케

class NN:
    def __init__(self):
        self.action_size = 32

        self.model = self.build_model()
        self.models = self.build_models()






    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu',
                         input_shape=(4, 4, 1)))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    def build_models(self):

        return [self.build_model() for __ in range(8)]


    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self, index):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = (self.models[index]).output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        # lr = 0.00025
        optimizer = RMSprop(lr=0.001, epsilon=0.01)
        updates = optimizer.get_updates( (self.models[index]).trainable_weights, [], loss)
        train = K.function([(self.models[index]).input, a, y], [loss], updates=updates)

        return train



if __name__ == "__main__":


    with open('memory.txt', 'rb') as f:
        memorys = pickle.load(f)
    print(len(memorys[1]))

