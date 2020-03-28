
import os
import sys
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import SGD


print(tf.__version__)
print(sys.version_info)


HEIGHT = 160
WIDTH = 160

BATCH_SIZE = 40

SIZE = HEIGHT

NUM_TRAIN = 185300  # 17786
NUM_VAL = 1981


# 数据准备
IM_WIDTH, IM_HEIGHT = 160, 160  # InceptionV3指定的图片尺寸
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量
nb_classes = 8




model = tf.keras.applications.InceptionResNetV2(weights=None,classes=nb_classes,input_shape=(HEIGHT, WIDTH, 3))
#model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,classes=nb_classes,input_shape=(HEIGHT, WIDTH, 3))


model.compile(
  optimizer=tf.keras.optimizers.RMSprop(),
  loss='categorical_crossentropy',
  metrics=['accuracy'])




model.summary()

datagen_train = ImageDataGenerator(
        rescale=1./255.0,
        rotation_range=1.5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

datagen_val = ImageDataGenerator(
        rescale=1./255.0)


train_generator=datagen_train.flow_from_directory('/home/nfs/em1/lgx/dataset/big_data',#类别子文件夹的上一级文件夹
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                        target_size=[SIZE, SIZE],
                                        class_mode='categorical'
                                      )

valid_generator=datagen_val.flow_from_directory('/home/nfs/em1/lgx/dataset/age_gender1/test',#类别子文件夹的上一级文件夹
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                        target_size=[SIZE, SIZE],
                                        class_mode='categorical'
                                      )

print(train_generator.class_indices)
print(valid_generator.class_indices)


epochs = 10000
filepath = "./model/inception-keras_model_{epoch:03d}-{val_acc:.4f}.h5" #避免文件名称重复
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,
                             save_best_only=False, mode='max')
history = model.fit_generator(generator = train_generator,
                           steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
                           epochs=epochs,
                           validation_data=valid_generator,
                              validation_steps=NUM_VAL // BATCH_SIZE,
                              verbose=1,callbacks=[checkpoint])