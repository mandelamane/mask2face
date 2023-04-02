import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.losses import mean_squared_error


class UNet:
    def __init__(self, input_size):
        inputs = Input(input_size)

        conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
        pool4 = MaxPooling2D((2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
        pool5 = MaxPooling2D((2, 2))(conv5)

        conv6 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool5)
        pool6 = MaxPooling2D((2, 2))(conv6)

        mid = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool6)

        up_conv1 = Conv2DTranspose(
            1024, (2, 2), strides=(2, 2), padding="same"
        )(mid)
        cat1 = concatenate([up_conv1, conv6])
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(cat1)

        up_conv2 = Conv2DTranspose(
            512, (2, 2), strides=(2, 2), padding="same"
        )(conv7)
        cat2 = concatenate([up_conv2, conv5])
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(cat2)

        up_conv3 = Conv2DTranspose(
            256, (2, 2), strides=(2, 2), padding="same"
        )(conv8)
        cat3 = concatenate([up_conv3, conv4])
        conv9 = Conv2D(256, (3, 3), activation="relu", padding="same")(cat3)

        up_conv4 = Conv2DTranspose(
            128, (2, 2), strides=(2, 2), padding="same"
        )(conv9)
        cat4 = concatenate([up_conv4, conv3])
        conv10 = Conv2D(128, (3, 3), activation="relu", padding="same")(cat4)

        up_conv5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(
            conv10
        )
        cat5 = concatenate([up_conv5, conv2])
        conv11 = Conv2D(64, (3, 3), activation="relu", padding="same")(cat5)

        up_conv6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(
            conv11
        )
        cat6 = concatenate([up_conv6, conv1])
        conv12 = Conv2D(32, (3, 3), activation="relu", padding="same")(cat6)

        outputs = Conv2D(3, (1, 1), activation="sigmoid")(conv12)

        self.model = Model(inputs=[inputs], outputs=[outputs])

    def get_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

    def save_summary(self, file_path):
        print("save architecture figure")
        tf.keras.utils.plot_model(
            self.model, show_shapes=True, to_file=file_path
        )

    def compile(self, learning_rate):
        self.model.compile(
            loss=UNet.ssim_l1_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=[UNet.ssim_l1_loss],
        )

    def fit(self, x, y, epochs, batch_size, callbacks):
        self.model.fit(
            x,
            y,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

    @staticmethod
    @tf.function
    def ssim_l1_loss(gt, y_pred, max_val=1.0, l1_weight=1.0):
        ssim_loss = 1 - tf.reduce_mean(
            tf.image.ssim(gt, y_pred, max_val=max_val)
        )
        l1 = mean_squared_error(gt, y_pred)
        return ssim_loss + tf.cast(l1 * l1_weight, tf.float32)

    @staticmethod
    @tf.function
    def ssim_loss(gt, y_pred, max_val=1.0):
        return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))
