import argparse
import os
from PIL import Image
import h5py
import cv2
import numpy as np
from tqdm import tqdm


def convert_imgfolder_to_h5(folder_path: str, h5_path: str):
    """Converts image folder to a h5 dataset.
    Args:
        folder_path (str): path to the image folder.
        h5_path (str): output path of the h5 file.
    """

    with h5py.File(h5_path, "w") as h5:
        classes = os.listdir(folder_path)
        # for class_name in tqdm(classes, desc="Processing classes"):
        for class_name in classes:
            cur_folder = os.path.join(folder_path, class_name)
            class_group = h5.create_group(class_name)
            for i, img_name in tqdm(enumerate(os.listdir(cur_folder)), desc="Processing Images"):
                data = Image.open(os.path.join(cur_folder, img_name))
                data = data.resize((224, 224), resample=Image.BICUBIC)
                data = np.asarray(data).astype("uint8")
                class_group.create_dataset(
                    img_name,
                    data=data,
                    shape=data.shape,
                    compression="gzip",
                    compression_opts=9,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    args = parser.parse_args()
    convert_imgfolder_to_h5(args.folder_path, args.h5_path)