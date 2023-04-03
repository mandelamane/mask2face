import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    DepthwiseConv2D,
)
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.utils import CustomObjectScope


def FReLU(inputs, kernel_size=3):
    x = DepthwiseConv2D(
        kernel_size, strides=(1, 1), padding="same", use_bias=False
    )(inputs)
    x = BatchNormalization()(x)
    x = tf.maximum(inputs, x)
    return x


def Mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


class UNet:
    def __init__(self, input_size):
        # エンコーダー側のフィルター数とカーネルサイズを定義
        filters = (64, 128, 128, 256, 256, 512)
        kernels = (7, 7, 7, 4, 3, 2)

        inputs = Input(input_size)

        # エンコーダー側の畳み込み層の出力を保持するリスト
        conv_outputs = []
        first_layer = Conv2D(filters[0], kernels[0], padding="same")(inputs)
        int_layer = first_layer

        for i, fil in enumerate(filters):
            int_layer, skip = UNet.down_block(int_layer, fil, kernels[i])
            conv_outputs.append(skip)

        int_layer = UNet.bottleneck(int_layer, filters[-1], kernels[-1])

        conv_outputs = list(reversed(conv_outputs))
        reversed_filters = list(reversed(filters))
        reversed_kernels = list(reversed(kernels))

        for i, fil in enumerate(reversed_filters):
            if i + 1 < len(reversed_filters):
                num_filter_next = reversed_filters[i + 1]
                num_kernel_next = reversed_kernels[i + 1]
            else:
                num_filter_next = fil
                num_kernel_next = reversed_kernels[i]

            int_layer = UNet.up_block(
                int_layer,
                conv_outputs[i],
                fil,
                num_filter_next,
                num_kernel_next,
            )

        int_layer = Concatenate()([first_layer, int_layer])
        int_layer = Conv2D(filters[0], kernels[0], padding="same")(int_layer)
        int_layer = UNet.activation(int_layer, "Mish")
        outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(
            int_layer
        )
        self.model = Model(inputs=[inputs], outputs=[outputs])

    @staticmethod
    def down_block(x, num_filter, kernel):
        x = Conv2D(num_filter, kernel, padding="same", strides=2)(x)
        out = Conv2D(num_filter, kernel, padding="same")(x)
        out = UNet.activation(out, "Mish")
        out = Conv2D(num_filter, kernel, padding="same")(out)

        out = Add()([out, x])
        return UNet.activation(out, "Mish"), x

    @staticmethod
    def bottleneck(x, num_filter, kernel):
        x = Conv2D(num_filter, kernel, padding="same")(x)
        return UNet.activation(x, "Mish")

    @staticmethod
    def up_block(x, skip, num_filter, num_filter_next, kernel):
        concat = Concatenate()([x, skip])

        out = Conv2D(num_filter, kernel, padding="same")(concat)
        out = UNet.activation(out, "Mish")
        out = Conv2D(num_filter, kernel, padding="same")(out)

        out = Add()([out, x])
        out = UNet.activation(out, "Mish")

        concat = Concatenate()([out, skip])

        out = Conv2DTranspose(
            num_filter_next, kernel, padding="same", strides=2
        )(concat)
        out = Conv2D(num_filter_next, kernel, padding="same")(out)
        return UNet.activation(out, "Mish")

    @staticmethod
    def activation(x, func):
        if func == "relu":
            x = Activation("relu")(x)
        elif func == "mish":
            x = Mish(x)
        elif func == "frelu":
            x = FReLU(x)

        return x

    def get_model(self, file_path):
        with CustomObjectScope(
            {"ssim_loss": UNet.ssim_loss, "ssim_l1_loss": UNet.ssim_l1_loss}
        ):
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
