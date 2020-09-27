# -*- coding: utf-8 -*-
# Group: MI
# Author: zhao chen
# Date: 2020-09-26

import numpy as np
import pandas as pd
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.random.set_seed(123)

with tf.device("cpu"):
    a = tf.constant([1])

a.device

b = tf.constant([0, 1])
tf.cast(b, dtype=tf.bool)

a = tf.range(5)
a

b = tf.Variable(a)
b.dtype
b.name
b.trainable

isinstance(b, tf.Tensor)
isinstance(b, tf.Variable)

tf.is_tensor(b)

b.numpy()


'''
创建tensor
from numpy, list
zeros, ones
fill
random
constant
application
'''
tf.convert_to_tensor()

tf.fill()

tf.random.normal()
tf.random.truncated_normal()

tf.random.uniform()

idx = tf.range(10)
idx = tf.random.shuffle(idx)

a = tf.random.normal([10, 784])
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)

a = tf.gather(a, idx)
b = tf.gather(b, idx)


'''
索引
# start:end:step
'''
a = tf.ones([1, 5, 5, 3])
a[:, :, :, 1]

a = tf.range(10)
a[-1:]
a[-2:]
a[::-1]
a[::-2]

a = tf.random.normal([2, 4, 28, 28, 3])
a.shape
a[0, ...].shape # ... 代替:
a[..., 0].shape
a[0, ..., 2].shape
a[1, 0, ..., 0].shape

'''
# selective indexing
tf.gather
tf.gather_nd
tf.boolean_mask
'''
a = tf.random.uniform([4, 35, 8], minval=0, maxval=100)
tf.gather(a, axis=0, indices=[2,3]).shape
a[2:4].shape
tf.gather(a, axis=0, indices=[2,1,3,0]).shape
tf.gather(a, axis=1, indices=[2, 3, 7]).shape
# axis 采样的不同纬度
# tf.gather_nd 不同纬度同时采样
tf.gather_nd(a, [0]).shape
tf.gather_nd(a, [0, 1]).shape
tf.gather_nd(a, [0, 1, 2]).shape #scalar
tf.gather_nd(a, [[0, 1, 2]]).shape

tf.gather_nd(a, [[0,0], [1,1]]).shape # [2, 8]
tf.gather_nd(a, [[0,0], [1,1], [2,2]]).shape # [3, 8]
tf.gather_nd(a, [[0,0,0], [1,1,1], [2,2,2]]).shape # [3]
tf.gather_nd(a, [[[0,0,0], [1,1,1], [2,2,2]]]).shape # [1, 3]

tf.boolean_mask(a, mask=[True, True, False, False]).shape


'''
纬度变换
shape, ndim
reshape

expand_dims/squeeze 加减纬度
transpose 转置/调整纬度
broadcast_to
'''

a = tf.random.normal([4, 28, 28, 3])
a.shape, a.ndim
tf.reshape(a, [4, -1, 3]).shape
tf.reshape(a, [4, -1]).shape

tf.transpose(a).shape
tf.transpose(a, perm=[0,1,3,2]).shape

# 增加纬度
tf.expand_dims(a, axis=0).shape
tf.expand_dims(a, axis=3).shape
tf.expand_dims(a, axis=-1).shape
# 减少纬度 仅对shape=1的dim
a = tf.random.normal([1,3,1,2])
tf.squeeze(a, axis=0).shape
tf.squeeze(a, axis=2).shape

'''
broadcasting 没有复制数据 
<- match from last dim ->
tf.tail 重复复制数据 占的内存较大
'''
x = tf.random.normal([4, 32, 32, 3])
(x + tf.random.normal([3])).shape # 1, 1, 1, 3 -> 4, 32, 32, 3
(x + tf.random.normal([32, 32, 1])).shape # 1, 32, 32, 1 -> 4, 32, 32, 3
(x + tf.random.normal([4, 1, 1 , 1])).shape # 4, 1, 1, 1 -> 4, 32, 32, 3
(x + tf.random.normal([1, 4, 1, 1])).shape # invalid


'''
数学运算
+-*/   element-wise
**,pow,square
sqrt
//, %
exp, log  tf.math.log  tf.exp 底为e 没有底为2
@ matmul
reduce_mean/max/min/sum  dim-wise
'''


'''
tf.concat
tf.stack

tf.unstack  axis的每个都打散
tf.split  指定打散都个数

'''

'''
tf.norm
tf.reduce_min/max
tf.argmax/argmin
tf.equal
tf.unique 返回tuple
'''

'''
排序
sort/argsort
topk  tf.math.top_k(a,k)
top-5 acc
'''
a = tf.random.shuffle(tf.range(5))
tf.sort(a, direction='DESCENDING', axis=-1) # 默认最后一个轴

idx = tf.argsort(a, direction="DESCENDING")
tf.gather(a, idx)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)

    return res


'''
数据的填充与复制
tf.pad
tf.title
tf.broadcast_to 隐式复制
'''
a = tf.random.normal([4, 28, 28, 3])
b = tf.pad(a, [[0,0], [2,2],[2,2,], [0,0]]) # 一一对应纬度
b.shape

a = tf.random.normal([3,3])
tf.tile(a, [1, 2]).shape # 1为不复制，保持不变；


'''
张量的限幅 裁剪
clip_by_value   => tf.maximum  tf.minimum
relu
clip_by_norm =>  方向不变，等比例缩放
gradient clipping 
一般参数对norm在0-20之前算是比较正常
'''
grads = tf.Tensor([1,2])
new_grads, total_norm = tf.clip_by_global_norm(grads, 25) # global 对所有对参数,参数方向不变，等比例缩放
# optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2, ...]))



'''
tf.where 与 tf.boolean_mask tf.gather_nd 相结合
tf.scatter_nd(indices, updates, shape) 更新底板（全为0）
tf.meshgrid 生成数据点
'''

a = tf.random.normal([3,3])
mask = a > 0
tf.boolean_mask(a, mask)

indices = tf.where(mask) # 返回为T的位置 索引
tf.gather_nd(a, indices) # 根据索引 返回值

# tf.where(cond, x, y)


x = tf.linspace(0.0, 2*3.14, 500)
y = tf.linspace(0.0, 2*3.14, 500)
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y])

def func(x):
    return tf.math.sin(x[..., 0]) + tf.math.cos(x[..., 1])
z = func(points)

'''
import matplotlib.pyplot as plt
plt.figure("a")
plt.imshow(z, origin="lower", inerpolation="none")
plt.colorbar() 
等高线等
'''

'''
数据集等加载



'''

import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()