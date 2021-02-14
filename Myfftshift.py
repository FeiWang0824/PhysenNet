# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import ops

def fftshift(x, axes=None):
    x = ops.convert_to_tensor(x)

    axes = tuple(range(2))
    shift = [dim // 2 for dim in [int(x.shape[0]),int(x.shape[1])]]
    
    result = tf.manip.roll(x, shift, axes)
    result = tf.cast(result,dtype=tf.complex64)
    
    return result

def ifftshift(x):
    x = ops.convert_to_tensor(x)

    axes = tuple(range(2))
    shift = [-(dim // 2) for dim in [int(x.shape[0]),int(x.shape[1])]]
    
    result = tf.manip.roll(x, shift, axes)
    result = tf.cast(result,dtype=tf.complex64)
    
    return result
