import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, m_imgs, f_imgs, num_img=4):
        self.num_img = num_img
        self.m_imgs = m_imgs
        self.f_imgs = f_imgs
        self.sum_num = self.m_imgs.shape[0] - self.num_img
        self.time_now = datetime.datetime.now()

        self.model_dir = f"model/{self.time_now}"
        self.fig_dir = f"result/{self.time_now}"
        os.mkdir(self.model_dir)
        os.mkdir(self.fig_dir)

    def on_epoch_end(self, epoch, logs=None):
        init_index = epoch % self.sum_num

        self.model.gen_f.save(f"model/mask2face_{epoch+1}.h5")
        self.model.gen_m.save(f"model/face2mask_{epoch+1}.h5")
        self.model.dis_f.save(f"model/dis_face_{epoch+1}.h5")
        self.model.dis_m.save(f"model/dis_mask_{epoch+1}.h5")

        _, ax = plt.subplots(self.num_img, 4, figsize=(12, 12))

        for i, (f_img, m_img) in enumerate(
            zip(
                self.f_imgs[init_index : init_index + self.num_img],
                self.m_imgs[init_index : init_index + self.num_img],
            )
        ):
            prediction = self.model.gen_f(m_img[np.newaxis, :, :, :])[
                0
            ].numpy()
            prediction2 = self.model.gen_m(m_img[np.newaxis, :, :, :])[
                0
            ].numpy()
            gen_f_img = (prediction * 127.5 + 127.5).astype(np.uint8)
            gen_m_img = (prediction2 * 127.5 + 127.5).astype(np.uint8)
            true_img = (f_img * 127.5 + 127.5).astype(np.uint8)
            input_img = (m_img * 127.5 + 127.5).astype(np.uint8)

            ax[i, 0].imshow(input_img)
            ax[i, 1].imshow(gen_f_img)
            ax[i, 2].imshow(gen_m_img)
            ax[i, 3].imshow(true_img)
            ax[i, 0].set_title("Mask image")
            ax[i, 1].set_title("Mask2Face image")
            ax[i, 2].set_title("Mask2Mask image")
            ax[i, 3].set_title("Face image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")

        plt.savefig(f"result/cyclegan_{epoch+1}.png")
        plt.close()


class UNetMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, mask_test, face_test, n_examples=4):
        self.n_examples = n_examples
        self.mask_test = mask_test
        self.face_test = face_test
        self.n_all = int(mask_test.shape[0] - n_examples)

    def on_epoch_end(self, epoch, log=None):
        _, ax = plt.subplots(self.n_examples, 3, figsize=(7, 10))

        start = epoch % self.n_all

        for i, (mask_img, face_img) in enumerate(
            zip(self.mask_test[start:], self.face_test[start:])
        ):
            if i == self.n_examples:
                break
            else:
                mask_img = mask_img[np.newaxis, :, :, :]
                output_img = (
                    (self.model(mask_img)[0] * 255.0).numpy().astype("uint8")
                )
                input_img = (mask_img[0] * 255.0).astype("uint8")
                collect_img = (face_img * 255.0).astype("uint8")

                ax[i, 0].imshow(input_img)
                ax[i, 1].imshow(output_img)
                ax[i, 2].imshow(collect_img)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Output image")
                ax[i, 2].set_title("Collect image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")
                ax[i, 2].axis("off")

        plt.savefig(f"result/unet_{epoch+1}.png")
        plt.close()
