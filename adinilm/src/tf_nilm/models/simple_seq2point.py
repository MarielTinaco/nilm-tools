
import keras

def create_model(input_window_length):

        """Specifies the structure of a seq2point model using Keras' functional API.

        Returns:
        model (tensorflow.keras.Model): The uncompiled seq2point model.

        """
        input_layer = keras.layers.Input(shape=(input_window_length,))
        reshape_layer = keras.layers.Reshape((1, input_window_length, 1))(input_layer)
        conv_layer_1 = keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
        conv_layer_2 = keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
        conv_layer_3 = keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
        conv_layer_4 = keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
        conv_layer_5 = keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
        flatten_layer = keras.layers.Flatten()(conv_layer_5)
        label_layer = keras.layers.Dense(1024, activation="relu")(flatten_layer)
        output_layer_1 = keras.layers.Dense(2*5, activation="linear", name="y1_output")(label_layer)
        output_layer_2 = keras.layers.Dense(5*5, activation="linear", name="y2_output")(label_layer)

        model = keras.Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])
        return model

def create_resnet_model(input_window_length):
    
        init = keras.initializers.HeNormal(seed=None)

        input_layer = keras.layers.Input(shape=(input_window_length,))
        reshape_layer = keras.layers.Reshape((1, input_window_length, 1))(input_layer)
        
        conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(reshape_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv2)
        resid_1 = keras.layers.Add()([conv2, conv3])
        resid_1 = keras.layers.ReLU()(resid_1)
        resid_1 = keras.layers.BatchNormalization()(resid_1)

        conv4 = keras.layers.Conv2D(filters=48, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(resid_1)
        conv4 = keras.layers.BatchNormalization()(conv4)
        conv5 = keras.layers.Conv2D(filters=48, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv4)
        conv5 = keras.layers.BatchNormalization()(conv5)
        conv5 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv5)
        conv6 = keras.layers.Conv2D(filters=48, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv5)
        conv6 = keras.layers.BatchNormalization()(conv6)
        resid_2 = keras.layers.Add()([conv5, conv6])
        resid_2 = keras.layers.ReLU()(resid_2)
        resid_2 = keras.layers.BatchNormalization()(resid_2)

        conv7 = keras.layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(resid_2)
        conv7 = keras.layers.BatchNormalization()(conv7)
        conv8 = keras.layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv7)
        conv8 = keras.layers.BatchNormalization()(conv8)
        conv8 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv8)
        conv9 = keras.layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv8)
        conv9 = keras.layers.BatchNormalization()(conv9)
        resid_3 = keras.layers.Add()([conv8, conv9])
        resid_3 = keras.layers.ReLU()(resid_3)
        resid_3 = keras.layers.BatchNormalization()(resid_3)

        conv10 = keras.layers.Conv2D(filters=96, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(resid_3)
        conv10 = keras.layers.BatchNormalization()(conv10)
        conv11 = keras.layers.Conv2D(filters=96, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv10)
        conv11 = keras.layers.BatchNormalization()(conv11)
        conv11 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv11)
        conv12 = keras.layers.Conv2D(filters=96, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv11)
        conv12 = keras.layers.BatchNormalization()(conv12)
        resid_4 = keras.layers.Add()([conv11, conv12])
        resid_4 = keras.layers.ReLU()(resid_4)
        resid_4 = keras.layers.BatchNormalization()(resid_4)

        conv13 = keras.layers.Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(resid_4)
        conv13 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv13)
        conv13 = keras.layers.BatchNormalization()(conv13)

        conv14 = keras.layers.Conv2D(filters=96, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv13)
        conv14 = keras.layers.BatchNormalization()(conv14)

        conv15 = keras.layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                                        kernel_initializer=init)(conv14)
        conv15 = keras.layers.BatchNormalization()(conv15)

        flatten_layer = keras.layers.Flatten()(conv15)
        
        label_layer = keras.layers.Dense(256, activation="relu",
                                        kernel_initializer=init)(flatten_layer)
        output_layer_1 = keras.layers.Dense(2*5, activation="linear", name="y1_output")(label_layer)
        output_layer_2 = keras.layers.Dense(5*5, activation="linear", name="y2_output")(label_layer)

        model = keras.Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])
        return model
