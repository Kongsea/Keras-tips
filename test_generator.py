#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import applications, optimizers
from keras.callbacks import CSVLogger
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from utils import generator_from_csv, get_data

# dimensions of our images.
(img_height, img_width) = (192, 64)

batch_size = 16

model = applications.VGG19(weights='imagenet', include_top=False,
                           input_shape=(img_height, img_width, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(8, activation='softmax'))

# add the model on top of the convolutional base
model = Model(inputs=model.input, outputs=top_model(model.output))

for layer in model.layers[: 19]:
  layer.trainable = False

# Two methods:
# 1. Use sparse_categorial_crossentropy
# 2. if using categorical_crossentropy:
#      should use keras.utils.np_utils.to_categorical in generator to convert labels.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_lines, valid_lines = get_data('data.csv')

ntrain, nvalid = len(train_lines), len(valid_lines)

print("""
Training set: %d images.
Validation set: %d images.
""" % (ntrain, nvalid))


train_generator = generator_from_csv(
    train_lines, batch_size=batch_size, target_size=(img_height, img_width), train=True)
validation_generator = generator_from_csv(
    valid_lines, batch_size=1, target_size=(img_height, img_width), train=False)

nbatches_train, _ = divmod(ntrain, batch_size)
nbatches_valid, _ = divmod(nvalid, 1)

filepath = "diy-{val_loss:.3f}-{val_acc:.3f}.h5"
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=nbatches_train * 2,
    epochs=100,
    verbose=2,
    max_queue_size=50,
    validation_data=validation_generator,
    validation_steps=nbatches_valid,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', verbose=1, save_best_only=True),
    ],
    workers=1,
)

# Use multiple workers, should make sure threadsafe in generator.

model.save_weights('csv_gen.h5')
