import struct
from array import array
from typing import List, Tuple

import numpy as np


class MnistDataloader:
    def __init__(
        self,
        training_images_filepath: str,
        training_labels_filepath: str,
        test_images_filepath: str,
        test_labels_filepath: str,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def load_data(self, shape: str = "flat", one_hot_encode: bool = False) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray | int]],
        List[Tuple[np.ndarray, np.ndarray | int]],
    ]:
        """
        Load and return training and test data.

        Parameters:
            shape: 'flat' → returns images as (784, 1) vectors,
                   'image' → returns images as (28, 28) arrays.

        Returns:
            Tuple of (training_data, test_data), each a list of (image, label) pairs.
        """
        training_data = self.read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath,
            shape,
            one_hot_encode,
        )
        test_data = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath, shape, one_hot_encode
        )
        return training_data, test_data

    def read_images_labels(
        self,
        images_filepath: str,
        labels_filepath: str,
        shape: str,
        one_hot_encode: bool,
    ) -> List[Tuple[np.ndarray, np.ndarray | int]]:
        with open(labels_filepath, "rb") as lbl_file:
            magic, size = struct.unpack(">II", lbl_file.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid label file magic number: {magic}")
            labels = array("B", lbl_file.read())

        with open(images_filepath, "rb") as img_file:
            magic, size2, rows, cols = struct.unpack(">IIII", img_file.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid image file magic number: {magic}")
            if size != size2:
                raise ValueError("Image and label count mismatch.")
            image_data = array("B", img_file.read())

        images = []
        for i in range(size):
            start = i * rows * cols
            end = start + rows * cols
            img = (
                np.array(image_data[start:end], dtype=np.float32) / 255.0
            )  # Normalize to [0,1]

            if shape == "flat":
                img = img.reshape(rows * cols, 1)  # Flattened column vector
            elif shape == "image":
                img = img.reshape(rows, cols)  # 2D image array
            else:
                raise ValueError(f"Invalid shape argument: {shape}")

            images.append(img)

        if one_hot_encode:
            labels = [self.one_hot_encode(label) for label in labels]

        return list(zip(images, labels))

    def one_hot_encode(self, label, num_classes=10):
        vec = np.zeros((num_classes, 1))
        vec[label] = 1.0
        return vec
