import tensorflow as tf
import tensorflow_addons as tfa


def downsample(
    x,
    filters,
    activation,
    kernel_init,
    gamma_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_init,
        padding=padding,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

    if activation:
        x = activation(x)

    return x


def upsample(
    x,
    filters,
    activation,
    kernel_init,
    gamma_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):

    x = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_init,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

    if activation:
        x = activation(x)

    return x


def residual_block(
    x,
    activation,
    kernel_init,
    gamma_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    use_bias=False,
):

    xdim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)

    x = tf.keras.layers.Conv2D(
        xdim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_init,
        padding=padding,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)

    x = tf.keras.layers.Conv2D(
        xdim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_init,
        padding=padding,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = tf.keras.layers.add([input_tensor, x])

    return x


class CycleGan(tf.keras.models.Model):
    def __init__(
        self,
        img_size,
        gen_optim,
        dis_optim,
        kernel_init,
        gamma_init,
        lambda_cycle=5.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_m = Generator(
            img_size=img_size,
            kernel_init=kernel_init,
            gamma_init=gamma_init,
            name="gen_m",
        )
        self.gen_f = Generator(
            img_size=img_size,
            kernel_init=kernel_init,
            gamma_init=gamma_init,
            name="gen_f",
        )
        self.dis_m = Discriminator(
            img_size=img_size,
            kernel_init=kernel_init,
            gamma_init=gamma_init,
            name="dis_m",
        )
        self.dis_f = Discriminator(
            img_size=img_size,
            kernel_init=kernel_init,
            gamma_init=gamma_init,
            name="dis_f",
        )

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.gen_m_optim = gen_optim
        self.gen_f_optim = gen_optim
        self.dis_m_optim = dis_optim
        self.dis_f_optim = dis_optim
        self.cycle_loss_obj = tf.keras.losses.MeanAbsoluteError()
        self.id_loss_obj = tf.keras.losses.MeanAbsoluteError()

    def calc_gen_loss(self, dis_fake):
        gen_loss = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(dis_fake), dis_fake
        )
        return gen_loss

    def calc_dis_loss(self, dis_real, dis_fake):
        real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(dis_real), dis_real)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(
            tf.zeros_like(dis_fake), dis_fake
        )
        return (real_loss + fake_loss) * 0.5

    def compile(self):
        super(CycleGan, self).compile()
        self.gen_m.get_weights("face2mask_41.h5")
        self.gen_m = self.gen_m.model
        self.gen_f.get_weights("mask2face_41.h5")
        self.gen_f = self.gen_f.model
        self.dis_m.get_weights("dis_mask_41.h5")
        self.dis_m = self.dis_m.model
        self.dis_f.get_weights("dis_face_41.h5")
        self.dis_f = self.dis_f.model


    @tf.function
    def train_step(self, batch_data):
        real_face, real_mask = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_mask = self.gen_m(real_face, training=True)
            fake_face = self.gen_f(real_mask, training=True)

            cycled_f2m2f = self.gen_f(fake_mask, training=True)
            cycled_m2f2m = self.gen_m(fake_face, training=True)

            same_face = self.gen_f(real_face, training=True)
            same_mask = self.gen_m(real_mask, training=True)

            dis_real_face = self.dis_f(real_face, training=True)
            dis_fake_face = self.dis_f(fake_face, training=True)

            dis_real_mask = self.dis_m(real_mask, training=True)
            dis_fake_mask = self.dis_m(fake_mask, training=True)

            gen_m_loss = self.calc_gen_loss(dis_fake_mask)
            gen_f_loss = self.calc_gen_loss(dis_fake_face)

            cycle_loss_m = (
                self.cycle_loss_obj(real_mask, cycled_m2f2m) * self.lambda_cycle
            )
            cycle_loss_f = (
                self.cycle_loss_obj(real_face, cycled_f2m2f) * self.lambda_cycle
            )

            id_loss_m = (
                self.id_loss_obj(real_mask, same_mask)
                * self.lambda_cycle
                * self.lambda_identity
            )

            id_loss_f = (
                self.id_loss_obj(real_face, same_face)
                * self.lambda_cycle
                * self.lambda_identity
            )

            total_loss_m = gen_m_loss + cycle_loss_m + id_loss_m
            total_loss_f = gen_f_loss + cycle_loss_f + id_loss_f

            dis_f_loss = self.calc_dis_loss(dis_real_face, dis_fake_face)
            dis_m_loss = self.calc_dis_loss(dis_real_mask, dis_fake_mask)

        grads_m = tape.gradient(total_loss_m, self.gen_m.trainable_variables)
        grads_f = tape.gradient(total_loss_f, self.gen_f.trainable_variables)

        dis_f_grads = tape.gradient(dis_f_loss, self.dis_f.trainable_variables)
        dis_m_grads = tape.gradient(dis_m_loss, self.dis_m.trainable_variables)

        self.gen_m_optim.apply_gradients(zip(grads_m, self.gen_m.trainable_variables))

        self.gen_f_optim.apply_gradients(zip(grads_f, self.gen_f.trainable_variables))

        self.dis_f_optim.apply_gradients(
            zip(dis_f_grads, self.dis_f.trainable_variables)
        )

        self.dis_m_optim.apply_gradients(
            zip(dis_m_grads, self.dis_m.trainable_variables)
        )

        return {
            "gen_f": gen_f_loss,
            "gen_m": gen_m_loss,
            "cyc_f": cycle_loss_f,
            "cyc_m": cycle_loss_m,
            "D_f": dis_f_loss,
            "D_m": dis_m_loss,
        }


class Generator:
    def __init__(
        self,
        img_size,
        kernel_init,
        gamma_init,
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=9,
        num_upsample_blocks=2,
        name=None,
    ):

        inputs = tf.keras.layers.Input(shape=img_size, name=name + "_input")

        x = ReflectionPadding2D(padding=(3, 3))(inputs)
        x = tf.keras.layers.Conv2D(
            filters, (7, 7), kernel_initializer=kernel_init, use_bias=False
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = tf.keras.layers.Activation("relu")(x)

        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = downsample(
                x,
                filters=filters,
                kernel_init=kernel_init,
                gamma_init=gamma_init,
                activation=tf.keras.layers.Activation("relu"),
            )

        for _ in range(num_residual_blocks):
            x = residual_block(
                x,
                kernel_init=kernel_init,
                gamma_init=gamma_init,
                activation=tf.keras.layers.Activation("relu"),
            )

        for _ in range(num_upsample_blocks):
            filters //= 2
            x = upsample(
                x,
                filters,
                kernel_init=kernel_init,
                gamma_init=gamma_init,
                activation=tf.keras.layers.Activation("relu"),
            )

        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = tf.keras.layers.Conv2D(3, (7, 7), padding="valid")(x)
        x = tf.keras.layers.Activation("tanh")(x)

        self.model = tf.keras.models.Model(inputs, x, name=name)
        self.model.summary()

    def get_weights(self, file_name):
        self.model.load_weights(f"../models/{file_name}")


class Discriminator:
    def __init__(
        self,
        img_size,
        kernel_init,
        gamma_init,
        filters=64,
        name=None,
    ):
        inputs = tf.keras.layers.Input(shape=img_size, name=name + "_img_input")

        x = tf.keras.layers.Conv2D(
            filters,
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_init,
        )(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        num_filters = filters
        for num_downsample_block in range(3):
            num_filters *= 2
            if num_downsample_block < 2:
                x = downsample(
                    x,
                    filters=num_filters,
                    activation=tf.keras.layers.LeakyReLU(0.2),
                    kernel_init=kernel_init,
                    gamma_init=gamma_init,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                )
            else:
                x = downsample(
                    x,
                    filters=num_filters,
                    activation=tf.keras.layers.LeakyReLU(0.2),
                    kernel_init=kernel_init,
                    gamma_init=gamma_init,
                    kernel_size=(4, 4),
                    strides=(1, 1),
                )

        x = tf.keras.layers.Conv2D(
            1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_init
        )(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)
        self.model.summary()

    def get_weights(self, file_name):
        self.model.load_weights(f"../models/{file_name}")


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]

        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def get_config(self):
        config = {"padding": self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
