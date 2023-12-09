import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def parse_file(path: Path) -> tuple[NDArray, NDArray]:
    reader = csv.reader(open(path))

    data = list(reader)

    classes = map(lambda x: x[0], data[1:])
    images = map(lambda x: x[1:], data[1:])

    return np.array(list(images)).astype(np.uint8) / 255, np.array(list(classes)).astype(np.uint8)


def process_raw_data():
    path = Path(__file__).parent / "data"
    test_path = path / "fashion-mnist_test.csv"
    train_path = path / "fashion-mnist_train.csv"

    train, test = parse_file(train_path), parse_file(test_path)

    np.save(path / "x_train.npy", train[0])
    np.save(path / "y_train.npy", train[1])
    np.save(path / "x_test.npy", test[0])
    np.save(path / "y_test.npy", test[1])


def load_fashion_mnist() -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    path = Path(__file__).parent / "data"

    x_train = np.load(path / "x_train.npy")
    y_train = np.load(path / "y_train.npy")
    x_test = np.load(path / "x_test.npy")
    y_test = np.load(path / "y_test.npy")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    process_raw_data()
