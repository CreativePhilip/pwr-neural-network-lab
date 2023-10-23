from pathlib import Path

import numpy as np
import scipy
from scipy.io.matlab import loadmat

from numpy.typing import NDArray


def load_matlab(file: str):
    def _check_keys(matlab_data):
        for key in matlab_data:
            if isinstance(matlab_data[key], scipy.io.matlab.mat_struct):
                matlab_data[key] = _to_dict(matlab_data[key])

        return matlab_data

    def _to_dict(matlab_object):
        output_data = {}
        for name in matlab_object._fieldnames:
            elem = matlab_object.__dict__[name]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                output_data[name] = _to_dict(elem)
            else:
                output_data[name] = elem
        return output_data

    data = loadmat(file, struct_as_record=False, squeeze_me=True)
    data = _check_keys(data)

    x = data["affNISTdata"]["image"].transpose() / 255
    y = data["affNISTdata"]["label_int"]

    return x, y


def _path(page: int) -> Path:
    return Path(__file__).parent / f"training_and_validation_batches/{page}.mat"


def load_affnist(page: int | list[int]) -> tuple[NDArray, NDArray]:
    page_numbers = page if isinstance(page, list) else [page]
    page_paths = [_path(no) for no in page_numbers]
    data_pages = [load_matlab(str(page)) for page in page_paths]

    data_pages_x = [page[0] for page in data_pages]
    data_pages_y = [page[1] for page in data_pages]

    return np.concatenate(data_pages_x), np.concatenate(data_pages_y)
