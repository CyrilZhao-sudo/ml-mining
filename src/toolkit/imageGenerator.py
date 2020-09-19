# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/9/19


import cv2 as cv
import os
import glob
import numpy as np
import random
from tensorflow import keras
import keras.preprocessing.image
from tensorflow.python.keras.utils.data_utils import Sequence



class VideoFrameGenerator(Sequence):

    def __init__(self,
                 rescale=1 / 255.,
                 nbframe: int = 5,
                 classes: list = [],
                 batch_size: int = 16,
                 use_frame_cache: bool = False,
                 target_shape: tuple = (224, 224),
                 shuffle: bool = True,
                 transformation: keras.preprocessing.image.ImageDataGenerator = None,
                 split: float = None,
                 nb_channel: int = 3,
                 glob_pattern: str = './videos/{classname}/*.avi',
                 _validation_data: list = None):
        """ Create a generator that return batches of frames from video
        - nbframe: int, number of frame to return for each sequence
        - classes: list of str, classes to infer
        - batch_size: int, batch size for each loop
        - use_frame_cache: bool, use frame cache (may take a lot of memory for large dataset)
        - shape: tuple, target size of the frames
        - shuffle: bool, randomize files
        - transformation: ImageDataGenerator with transformations
        - split: float, factor to split files and validation
        - nb_channel: int, 1 or 3, to get grayscaled or RGB images
        - glob_pattern: string, directory path with '{classname}' inside that
            will be replaced by one of the class list
        - _validation_data: already filled list of data, do not touch !

        You may use the "classes" property to retrieve the class list afterward.

        The generator has that properties initialized:
        - classes_count: number of classes that the generator manages
        - files_count: number of video that the generator can provides
        - classes: the given class list
        - files: the full file list that the generator will use, this
            is usefull if you want to remove some files that should not be
            used by the generator.
        """

        # should be only RGB or Grayscale
        assert nb_channel in (1, 3)

        # we should have classes
        assert len(classes) > 0

        # shape size should be 2
        assert len(target_shape) == 2

        # split factor should be a propoer value
        if split is not None:
            assert split < 1.0 and split > 0.0

        # be sure that classes are well ordered
        classes.sort()

        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nbframe = nbframe
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation
        self.use_frame_cache = use_frame_cache

        self._random_trans = []
        self.__frame_cache = {}
        self.files = []
        self.validation = []

        if _validation_data is not None:
            # we only need to set files here
            self.files = _validation_data
        else:
            if split is not None and split > 0.0:
                for c in classes:
                    files = glob.glob(glob_pattern.format(classname=c))
                    nbval = int(split * len(files))

                    print("class %s, validation count: %d" % (c, nbval))

                    # generate validation indexes
                    indexes = np.arange(len(files))

                    if shuffle:
                        np.random.shuffle(indexes)

                    val = np.random.permutation(indexes)[:nbval]  # get some sample
                    indexes = np.array([i for i in indexes if i not in val])  # remove validation from train

                    # and now, make the file list
                    self.files += [files[i] for i in indexes]
                    self.validation += [files[i] for i in val]

            else:
                for c in classes:
                    self.files += glob.glob(glob_pattern.format(classname=c))

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        self.on_epoch_end()  # to initialize transformations and shuffle indices

        print("get %d classes for %d files for %s" % (
            self.classes_count,
            self.files_count,
            'train' if _validation_data is None else 'validation'))

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            nbframe=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            _validation_data=self.validation)

    def on_epoch_end(self):
        # prepare transformation to avoid __getitem__ to reinitialize them
        if self.transformation is not None:
            self._random_trans = []
            for i in range(self.files_count):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.files_count / self.batch_size))

    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        t = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                t = self._random_trans[i]

            # video = random.choice(files)
            video = self.files[i]
            cl = video.split(os.sep)[-2]

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(cl)
            label[col] = 1.

            if video not in self.__frame_cache:
                cap = cv.VideoCapture(video)
                frames = []
                while True:
                    grabbed, frame = cap.read()
                    if not grabbed:
                        # end of video
                        break

                        # resize
                    frame = cv.resize(frame, shape)

                    # use RGB or Grayscale ?
                    if self.nb_channel == 3:
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    else:
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                    # to np
                    frame = keras.preprocessing.image.img_to_array(frame) * self.rescale

                    # keep frame
                    frames.append(frame)

                # Add 2 frames to drop first and last frame
                jump = len(frames) // (nbframe + 2)

                # get only some images
                try:
                    frames = frames[jump::jump][:nbframe]
                except Exception as e:
                    print(video)
                    raise e

                # add to frame cache to not read from disk later
                if self.use_frame_cache:
                    self.__frame_cache[video] = frames
            else:
                frames = self.__frame_cache[video]

            # apply transformation
            if t is not None:
                frames = [self.transformation.apply_transform(frame, t) for frame in frames]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)


# coding=utf-8
'''
Created on 2018-7-10

'''
import keras
import math
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            # x_train数据
            image = cv2.imread(data)
            image = list(image)
            images.append(image)
            # y_train数据
            right = data.rfind("\\", 0)
            left = data.rfind("\\", 0, right) + 1
            class_name = data[left:right]
            if class_name == "dog":
                labels.append([0, 1])
            else:
                labels.append([1, 0])
        # 如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return np.array(images), np.array(labels)


# 读取样本名称，然后根据样本名称去读取数据
class_num = 0
train_datas = []
for file in os.listdir("D:/xxx"):
    file_path = os.path.join("D:/xxx", file)
    if os.path.isdir(file_path):
        class_num = class_num + 1
        for sub_file in os.listdir(file_path):
            train_datas.append(os.path.join(file_path, sub_file))

# 数据生成器
training_generator = DataGenerator(train_datas)

# 构建网络
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=784))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(training_generator, epochs=50, max_queue_size=10, workers=1)
