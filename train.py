import argparse
import os

import numpy as np
import tensorflow as tf

from cyclegan import CycleGan
from monitor import GANMonitor, UNetMonitor
from unet import UNet

tf_autotune = tf.data.AUTOTUNE


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_dataset", "-fd", help="face dataset name")
    parser.add_argument("--mask_dataset", "-md", help="mask dataset name")
    parser.add_argument(
        "--model",
        "-m",
        default="unet",
        type=str,
        help="model architecture [cyclegan, unet]",
    )
    parser.add_argument(
        "--pre_trained",
        "-p",
        default="none",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--batch_size", "-b", default=12, type=int, help="batch size"
    )
    parser.add_argument(
        "--epochs", "-e", default=20, type=int, help="train epochs"
    )
    parser.add_argument(
        "--learning_rate", "-lr", default=1e-4, type=int, help="learning rate"
    )
    args = parser.parse_args()
    return args


def unnpz(file_path):
    images = np.load(file_path)["img"]
    return images


def load_data(file_names, test_data_rate):
    datasets = []
    for file_name in file_names:
        images = unnpz(f"datasets/{file_name}.npz")
        datasets.append(images)

    face, mask = datasets[0], datasets[1]
    face = face.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    train_num = int(face.shape[0] * (1 - test_data_rate))
    f_train, f_test = face[:train_num], face[train_num:]
    m_train, m_test = mask[:train_num], mask[train_num:]

    return f_train, m_train, f_test, m_test


def train_cyclegan(args):
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    f_train, m_train, f_test, m_test = load_data(
        [args.face_dataset, args.mask_dataset], 0.1
    )
    img_size = f_train.shape[1:]

    gen_optim = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, beta_1=0.5
    )
    dis_optim = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, beta_1=0.5
    )
    cycle_gan = CycleGan(
        img_size=img_size,
        gen_optim=gen_optim,
        dis_optim=dis_optim,
        kernel_init=kernel_init,
        gamma_init=gamma_init,
    )

    cycle_gan.compile()
    plotter = GANMonitor(m_test, f_test)

    cycle_gan.fit(
        f_train,
        m_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[plotter],
    )


def train_unet(args):
    f_train, m_train, f_test, m_test = load_data(
        [args.face_dataset, args.mask_dataset], 0.1
    )

    img_size = f_train.shape[1:]

    unet = UNet(img_size)

    if args.pre_trained != "none":
        unet.get_model(os.path.join("model", f"{args.pre_trained}.h5"))

    unet.compile(args.learning_rate)
    unet.summary()
    unet.save_summary(os.path.join("model", "unet_arch.png"))

    plotter = UNetMonitor(m_test, f_test)
    checkpoint_filepath = "model/unet_{epoch}.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, use_multiprocessing=True
    )

    unet.fit(
        m_train,
        f_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[plotter, model_checkpoint_callback],
    )


def main(args):
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

    if args.model == "cyclegan":
        train_cyclegan(args)
    elif args.model == "unet":
        train_unet(args)
    else:
        print("--model is either cyclegan or unet")


if __name__ == "__main__":
    args = read_args()
    main(args)
