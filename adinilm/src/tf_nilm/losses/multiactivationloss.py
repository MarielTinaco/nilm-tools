
import numpy as np
import tensorflow as tf

class MultiActivationLoss(tf.keras.losses.SparseCategoricalCrossentropy):

	def call(self, y_true, y_pred):

		pred_state = tf.reshape(y_pred, [-1, 5, 2])

		pred_state_softmax = tf.nn.softmax(pred_state, axis=1)

		loss_nll = super().call(y_true, pred_state_softmax)

		return loss_nll