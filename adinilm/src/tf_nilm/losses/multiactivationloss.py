
import numpy as np
import tensorflow as tf

class MultiActivationLoss(tf.keras.losses.BinaryCrossentropy):

	def call(self, y_true, y_pred, **kwargs):
		y_true = tf.cast(y_true, dtype=np.uint8)
		true_state = tf.one_hot(y_true, depth=2)
		return super().call(true_state, y_pred, **kwargs)