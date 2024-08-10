
import logging
import tensorflow as tf


class PyLoggingCallback(tf.keras.callbacks.Callback):

        def __init__(self, *args, **kwargs):
                super().__init__()
                logging.basicConfig(*args, **kwargs)

        def on_train_begin(self, logs=None):
                logging.info(f"Train params: {self.params}")
                logging.info(f"Train logs: {logs}")

        def on_epoch_begin(self, epoch, logs=None):
                logging.info(f"Epoch {epoch} begin: {logs}")

        def on_epoch_end(self, epoch, logs=None):
                logging.info(f"Epoch {epoch} end: {logs}")