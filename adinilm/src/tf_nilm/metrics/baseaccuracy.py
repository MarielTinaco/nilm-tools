
import tensorflow as tf


class BaseAccuracy(tf.keras.metrics.Accuracy):
    
        def update_state(self, y_true, y_pred, sample_weight=None):
                y_pred = tf.argmax(y_pred, axis=2)
                return super().update_state(y_true, y_pred, sample_weight)