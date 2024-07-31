
import numpy as np
import tensorflow as tf

class QuantileLoss(tf.keras.Loss):
	def __init__(self, quantiles=[0.0025,0.1, 0.5, 0.9, 0.975], *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.quantiles = quantiles

	def call(self, y_true, y_pred):
		pred_rms = tf.reshape(y_pred, [-1, 5, 5])

		targets = tf.expand_dims(y_true, axis=1)
		targets = tf.repeat(targets, repeats=[5], axis=1)

		quantiles = tf.convert_to_tensor(self.quantiles, dtype=tf.float32)

		error = tf.transpose((targets - pred_rms), perm=[0,2,1])
		loss = tf.math.maximum(quantiles*error, (quantiles-1)*error)
		return tf.reduce_mean(loss, axis=1)