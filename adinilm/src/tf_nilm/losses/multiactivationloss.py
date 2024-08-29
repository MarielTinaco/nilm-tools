
import numpy as np
import tensorflow as tf

class MultiActivationLoss(tf.keras.losses.BinaryCrossentropy):

	def call(self, y_true, y_pred):
		y_true = tf.cast(y_true, dtype=np.uint8)

		pred_state = tf.reshape(y_pred, [-1, 5, 2])
		true_state = tf.one_hot(y_true, depth=2)

		# pred_state_softmax = tf.nn.softmax(pred_state, axis=2)
		# pred_state_argmax = tf.argmax(pred_state_softmax, axis=2)

		# pred_state_softmax = tf.cast(pred_state_softmax, np.float64)
		loss_nll = super().call(true_state, pred_state)

		return loss_nll