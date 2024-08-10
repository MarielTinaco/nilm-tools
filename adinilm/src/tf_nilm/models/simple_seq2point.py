
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