import argparse
import os

import pandas as pd
import tensorflow as tf

from cyclegan import CycleGan
from data_generator import ImageDataGenerator
from monitor import GANMonitor, UNetMonitor
from unet import UNet

tf_autotune = tf.data.AUTOTUNE


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_input_dir",
        "-tid",
        type=str,
        help="Training input dataset directory",
    )
    parser.add_argument(
        "--train_target_dir",
        "-ttd",
        type=str,
        help="Training target dataset directory",
    )
    parser.add_argument(
        "--val_input_dir",
        "-vid",
        type=str,
        help="Validation input dataset directory",
    )
    parser.add_argument(
        "--val_target_dir",
        "-vtd",
        type=str,
        help="Validation target dataset directory",
    )
    parser.add_argument(
        "--img_size",
        "-is",
        nargs=2,
        type=int,
        help="Image size (width and height)",
    )
    parser.add_argument(
        "--batch_size", "-b", default=12, type=int, help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        "-e",
        default=20,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=1e-4,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="unet",
        type=str,
        help="Model architecture [cyclegan, unet]",
    )
    parser.add_argument(
        "--pre_trained",
        "-p",
        default="none",
        type=str,
        help="Pretrained model path",
    )

    args = parser.parse_args()
    return args


def train_cyclegan(args):
    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate

    train_generator = ImageDataGenerator(
        args.train_input_dir,
        args.train_target_dir,
        batch_size,
        img_size,
    )
    val_generator = ImageDataGenerator(
        args.val_input_dir, args.val_target_dir, batch_size, img_size
    )
    test_generator = ImageDataGenerator(
        args.val_input_dir, args.val_target_dir, batch_size, img_size
    )

    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    img_size = tuple(args.img_size) + (3,)

    gen_optim = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.5)
    dis_optim = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.5)

    cycle_gan = CycleGan(
        img_size=img_size,
        gen_optim=gen_optim,
        dis_optim=dis_optim,
        kernel_init=kernel_init,
        gamma_init=gamma_init,
    )

    cycle_gan.compile()
    plotter = GANMonitor(test_generator)

    cycle_gan.summary()
    cycle_gan.save_summary("model")

    history = cycle_gan.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[plotter],
    )

    return history


def train_unet(args):
    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate

    train_generator = ImageDataGenerator(
        args.train_input_dir,
        args.train_target_dir,
        batch_size,
        img_size,
    )
    val_generator = ImageDataGenerator(
        args.val_input_dir, args.val_target_dir, batch_size, img_size
    )
    test_generator = ImageDataGenerator(
        args.val_input_dir, args.val_target_dir, batch_size, img_size
    )

    img_size = tuple(args.img_size) + (3,)
    unet = UNet(img_size)

    if args.pre_trained != "none":
        unet.load_model(os.path.join("model", f"{args.pre_trained}.h5"))

    unet.compile(lr)
    unet.summary()
    unet.save_summary(os.path.join("model", "unet_arch.png"))

    plotter = UNetMonitor(test_generator)
    checkpoint_filepath = "model/unet_{epoch}.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, use_multiprocessing=True
    )

    history = unet.fit(
        train_generator,
        val_generator,
        epochs,
        [plotter, model_checkpoint_callback],
    )

    return history


def optimize_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(
                "{} memory growth: {}".format(
                    device, tf.config.experimental.get_memory_growth(device)
                )
            )
    else:
        print("Not enough GPU hardware devices available")


def main(args):
    optimize_gpu()

    if args.model == "cyclegan":
        history = train_cyclegan(args)
    elif args.model == "unet":
        history = train_unet(args)
    else:
        print("--model is either cyclegan or unet")
        exit()

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv("history.csv")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
