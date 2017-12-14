#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import threading

import keras
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def get_data(csv_file):
  with open(csv_file, 'rb') as f:
    lines = [line.strip() for line in f.readlines()]

  train_lines = []
  valid_lines = []
  for line in lines:
    if 'train' in line:
      train_lines.append(line)
    elif 'validation' in line:
      valid_lines.append(line)

  return train_lines, valid_lines


class threadsafe_iter(object):
  """Takes an iterator/generator and makes it thread-safe by
  serializing call to the `next` method of given iterator/generator.
  """

  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def next(self):
    with self.lock:
      return self.it.next()


def threadsafe_generator(f):
  """A decorator that takes a generator function and makes it thread-safe.
  """
  def g(*a, **kw):
    return threadsafe_iter(f(*a, **kw))
  return g


@threadsafe_generator
def generator_from_csv(lines, batch_size, target_size, train=True):

  nbatches, _ = divmod(len(lines), batch_size)

  cls = {'香蕉': 7, '芒果': 2, '原味': 0, '菠萝': 5, '椰子': 1, '草莓': 4, '蜜桃': 6, '苹果': 3}

  count = 1
  epoch = 0

  while 1:
    np.random.shuffle(lines)
    epoch += 1
    i, j = 0, batch_size

    # Mini-batches within epoch.
    mini_batches_completed = 0

    for _ in range(nbatches):
      sub = lines[i:j]
      try:
        X = np.array([(2 * (img_to_array(load_img(f, target_size=target_size)) / 255.0 - 0.5))
                      for f in sub])

        if train:
          Y = keras.utils.np_utils.to_categorical(
              np.array([cls[s[19 + 7 * 3:19 + 9 * 3]] for s in sub]), 8)
        else:
          Y = keras.utils.np_utils.to_categorical(
              np.array([cls[s[24 + 7 * 3:24 + 9 * 3]] for s in sub]), 8)

        mini_batches_completed += 1
        yield X, Y

      except IOError:
        count -= 1

      i = j
      j += batch_size
      count += 1
