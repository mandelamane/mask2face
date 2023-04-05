import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def save_images(ax, mask_img, gen_face_img, face_img, gen_mask_img=None):
    ax[0].imshow(mask_img)
    ax[1].imshow(gen_face_img)
    ax[2].imshow(face_img)
    ax[0].set_title("Mask image")
    ax[1].set_title("Generated Face image")
    ax[2].set_title("Face image")
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    if gen_mask_img is not None:
        ax[3].imshow(gen_mask_img)
        ax[3].set_title("Generated Mask image")
        ax[3].axis("off")


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, test_data_generator, n_examples=4):
        self.n_examples = n_examples
        self.test_data_generator = test_data_generator

    def on_epoch_end(self, epoch, logs=None):
        self.model.gen_f.save(f"model/mask2face_{epoch+1}.h5")
        self.model.gen_m.save(f"model/face2mask_{epoch+1}.h5")
        self.model.dis_f.save(f"model/dis_face_{epoch+1}.h5")
        self.model.dis_m.save(f"model/dis_mask_{epoch+1}.h5")

        _, ax = plt.subplots(self.num_img, 4, figsize=(12, 12))

        for i in range(self.n_examples):
            mask_img, face_img = self.test_data_generator[i]

            prediction = self.model.gen_f(mask_img[np.newaxis, :, :, :])[
                0
            ].numpy()
            prediction2 = self.model.gen_m(face_img[np.newaxis, :, :, :])[
                0
            ].numpy()
            gen_face_img = (prediction * 255.0).astype(np.uint8)
            gen_mask_img = (prediction2 * 255.0).astype(np.uint8)
            face_img = (face_img * 255.0).astype(np.uint8)
            mask_img = (mask_img * 255.0).astype(np.uint8)

            save_images(ax[i], mask_img, gen_face_img, face_img, gen_mask_img)

        plt.savefig(f"result/cyclegan_{epoch+1}.png")
        plt.close()


class UNetMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, test_data_generator, n_examples=4):
        self.n_examples = n_examples
        self.test_data_generator = test_data_generator

    def on_epoch_end(self, epoch, log=None):
        _, ax = plt.subplots(self.n_examples, 3, figsize=(7, 10))

        for i in range(self.n_examples):
            mask_img, face_img = self.test_data_generator[i]

            mask_img = mask_img[np.newaxis, :, :, :]
            gen_face_img = (
                (self.model(mask_img)[0] * 255.0).numpy().astype("uint8")
            )
            mask_img = (mask_img[0] * 255.0).astype("uint8")
            face_img = (face_img * 255.0).astype("uint8")

            save_images(ax[i], mask_img, gen_face_img, face_img)

        plt.savefig(f"result/unet_{epoch+1}.png")
        plt.close()
