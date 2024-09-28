
import tensorflow as tf
import numpy as np
import keras
# from keras.src.utils import metrics_utils
from keras.src import backend

from keras.metrics import MeanMetricWrapper


def get_eac(target, prediction):
	num = tf.reduce_sum(tf.abs(target - prediction), axis=0)
	den = 2*tf.reduce_sum(target, axis=0)
	return (1 - num/den)

def eac(y_true, y_pred):

	y_pred = y_pred[:,:,2]
	y_pred = tf.clip_by_value(y_pred, -128, 127)

	if y_true.dtype != y_pred.dtype:
		y_pred = tf.cast(y_pred, y_true.dtype)
	y_pred = (tf.clip_by_value(y_pred + 128, 0, 256.) / 256. ) + 1e-3
	y_true = ((y_true + 128) / 256) + 1e-3

	return tf.cast(tf.reduce_mean(get_eac(y_true, y_pred), axis=0) , backend.floatx())

@keras.saving.register_keras_serializable()
class EAC(MeanMetricWrapper):
    
    def __init__(self, name='eac', dtype=None):
        super().__init__(eac ,name, dtype)