#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:05:49 2018

@author: jessica
"""
#https://ithelp.ithome.com.tw/articles/10190971
import numpy as np
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.metrics import load_model
import itertools
import matplotlib.pyplot as plt
from PIL import Image

train_path = './train'
valid_path = './valid'
test_path = './test'

#ImageDataGenerator顧名思義就是用來產生圖片資料的：
#用以生成一個批次的圖像數據。訓練時該函數會無限生成數據，直到達到規定的epoch次數為止。
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['dogs', 'cats'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['dogs', 'cats'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['dogs', 'cats'], batch_size=10)

#flow_from_directory（）：
#以文件夾路徑為參數，生成經過數據提升/歸一化後的數據，在一個無限循環中無限制生產批次資料。
#classes：
#是可選參數，為子文件夾的列表，如上我們分別為['dogs'，'cats']的分類，默認為無。若未提供，則該類別列表將從目錄下的子文件夾名稱/結構自動推斷。每一個子文件夾都會被認為是一個新的類！

#target_size=(224,224)：
#整數元組默認為（256,256），圖像將被調整大小成該尺寸，因為我基於VGG16模型，所以這裡設定為(224,224)。

#Found 40 images belonging to 2 classes.
#Found 10 images belonging to 2 classes.
#Found 10 images belonging to 2 classes.
print(train_batches.image_shape)
# (224, 224, 3)

vgg16_model = keras.applications.vgg16.VGG16()
model = Sequential()
model.summary()

for layer in vgg16_model.layers:
    model.add(layer)
model.summary()

#將頂層predictions拿掉，基本上遷移式學習不只是刪掉原本輸出，不過因為資料類型的關係這裡只把最後一層刪掉重新訓練。
model.layers.pop()
for layer in model.layers:
    layer.trainable = False
    
model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=.00002122), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=10, validation_data=valid_batches, validation_steps=4, epochs=10, verbose=2)
model.save("cat-dog-model-base-VGG16.h5")