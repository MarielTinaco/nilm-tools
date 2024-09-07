
from tensorflow_model_optimization.python.core.keras.compat import keras

def create_resnet_model(input_window_length, dropout=0.2):
    
        init = keras.initializers.HeNormal(seed=None)

        input_layer = keras.layers.Input(shape=(input_window_length,))
        input_layer = keras.layers.Reshape((input_window_length,1))(input_layer)

        conv1 = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv2)
        resid_1 = keras.layers.Add()([conv2, conv3])
        resid_1 = keras.layers.ReLU()(resid_1)
        resid_1 = keras.layers.BatchNormalization()(resid_1)

        conv4 = keras.layers.Conv1D(filters=48, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(resid_1)
        conv4 = keras.layers.BatchNormalization()(conv4)
        conv5 = keras.layers.Conv1D(filters=48, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv4)
        conv5 = keras.layers.BatchNormalization()(conv5)
        conv5 = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv5)
        conv6 = keras.layers.Conv1D(filters=48, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv5)
        conv6 = keras.layers.BatchNormalization()(conv6)
        resid_2 = keras.layers.Add()([conv5, conv6])
        resid_2 = keras.layers.ReLU()(resid_2)
        resid_2 = keras.layers.BatchNormalization()(resid_2)

        dropout_layer_0 = keras.layers.Dropout(dropout)(resid_2)

        conv7 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(dropout_layer_0)
        conv7 = keras.layers.BatchNormalization()(conv7)
        conv8 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv7)
        conv8 = keras.layers.BatchNormalization()(conv8)
        conv8 = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv8)
        conv9 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv8)
        conv9 = keras.layers.BatchNormalization()(conv9)
        resid_3 = keras.layers.Add()([conv8, conv9])
        resid_3 = keras.layers.ReLU()(resid_3)
        resid_3 = keras.layers.BatchNormalization()(resid_3)

        dropout_layer_1 = keras.layers.Dropout(dropout)(resid_3)

        conv10 = keras.layers.Conv1D(filters=96, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(dropout_layer_1)
        conv10 = keras.layers.BatchNormalization()(conv10)
        conv11 = keras.layers.Conv1D(filters=96, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv10)
        conv11 = keras.layers.BatchNormalization()(conv11)
        conv11 = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv11)
        conv12 = keras.layers.Conv1D(filters=96, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv11)
        conv12 = keras.layers.BatchNormalization()(conv12)
        resid_4 = keras.layers.Add()([conv11, conv12])
        resid_4 = keras.layers.ReLU()(resid_4)
        resid_4 = keras.layers.BatchNormalization()(resid_4)

        dropout_layer_2 = keras.layers.Dropout(dropout)(resid_4)

        conv13 = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(dropout_layer_2)
        conv13 = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(conv13)
        conv13 = keras.layers.BatchNormalization()(conv13)

        conv14 = keras.layers.Conv1D(filters=96, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv13)
        conv14 = keras.layers.BatchNormalization()(conv14)

        conv15 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="causal", activation="relu",
                                        kernel_initializer=init)(conv14)
        conv15 = keras.layers.BatchNormalization()(conv15)

        flatten_layer = keras.layers.Flatten()(conv15)
        
        dropout_layer_3 = keras.layers.Dropout(dropout)(flatten_layer)

        label_layer = keras.layers.Dense(256, activation="linear",
                                        kernel_initializer=init)(dropout_layer_3)
        output_layer_1 = keras.layers.Dense(5*2, activation="sigmoid")(label_layer)
        output_layer_2 = keras.layers.Dense(5*5, activation="linear")(label_layer)

        output_layer_1 = keras.layers.Reshape((5, 2), name="y1_output")(output_layer_1)
        output_layer_2 = keras.layers.Reshape((5, 5), name="y2_output")(output_layer_2)

        model = keras.Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])
        return model
