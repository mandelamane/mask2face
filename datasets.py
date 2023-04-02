import argparse
import glob

import cv2
import numpy as np
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-id", nargs="+", type=str, help="images directory name"
    )
    parser.add_argument("--output", "-o", type=str, help="output file name")
    parser.add_argument(
        "--resize",
        "-r",
        type=int,
        help="size to resize image",
    )

    args = parser.parse_args()
    return args


def make_datasets(args):
    input_dir_names = args.input_dir
    save_file_name = args.output
    img_size = (args.resize, args.resize)

    preprocess_imgs(
        input_dir_names,
        save_file_name,
        img_size,
    )


def preprocess_imgs(
    input_dir_names,
    save_file_name,
    img_size,
):
    imgs = [
        cv2.imread(file_path)[:, :, ::-1]
        for dir_name in input_dir_names
        for file_path in glob.iglob(f"data/{dir_name}/*")
    ]

    dataset = np.array(
        [
            cv2.resize(img, img_size)
            for img in tqdm(
                imgs,
            )
        ]
    )

    save_file = f"datasets/{save_file_name}.npz"
    np.savez(save_file, img=dataset)
    print("datasets image size:", dataset.shape)
    print("save directory", save_file)


if __name__ == "__main__":
    args = read_args()
    make_datasets(args)
