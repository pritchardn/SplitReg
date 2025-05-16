"""
Contains implemented data loaders for various radio astronomy datasets.
"""
import os
import pickle
from typing import Union

import numpy as np

from data.utils import test_train_split, extract_polarization
from interfaces.data.raw_data_loader import RawDataLoader


def _normalize(
        image_data: np.ndarray, masks: np.ndarray, min_threshold: int, max_threshold: int
) -> np.ndarray:
    normalized_data = (image_data - np.mean(image_data, axis=0)) / np.std(image_data, axis=0)
    _max = np.mean(image_data[np.invert(masks)]) + max_threshold * np.std(
        image_data[np.invert(masks)]
    )
    _min = np.absolute(
        np.mean(image_data[np.invert(masks)])
        - min_threshold * np.std(image_data[np.invert(masks)])
    )
    image_data = np.clip(image_data, _min, _max)
    image_data = np.log(image_data)
    # Rescale
    minimum, maximum = np.min(image_data), np.max(image_data)
    image_data = (image_data - minimum) / (maximum - minimum)
    return image_data


class HeraDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.train_x[self.train_x == np.inf] = np.finfo(self.train_x.dtype).max
        self.test_x[self.test_x == np.inf] = np.finfo(self.test_x.dtype).max
        self.test_x = self.test_x.astype("float32")
        self.train_x = self.train_x.astype("float32")
        self.test_x = _normalize(self.test_x, self.test_y, 1, 4)
        self.train_x = _normalize(self.train_x, self.train_y, 1, 4)
        self.convert_pytorch()
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def load_data(self, excluded_rfi: Union[str, None] = None):
        if excluded_rfi is None:
            rfi_models = []
            file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all.pkl")
            data, _, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
        else:
            rfi_models = ["rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
            rfi_models.remove(excluded_rfi)
            test_file_path = os.path.join(
                self.data_dir, f"HERA_21-11-2024_{excluded_rfi}.pkl"
            )
            _, _, test_x, test_y = np.load(test_file_path, allow_pickle=True)

            train_file_path = os.path.join(
                self.data_dir, f'HERA_21-11-2024_{"-".join(rfi_models)}.pkl'
            )
            train_x, train_y, _, _ = np.load(train_file_path, allow_pickle=True)
        self.train_x = np.moveaxis(train_x, 1, 2)
        self.train_y = np.moveaxis(train_y, 1, 2)
        self.test_x = np.moveaxis(test_x, 1, 2)
        self.test_y = np.moveaxis(test_y, 1, 2)
        self.rfi_models = rfi_models
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        self.test_y = extract_polarization(self.test_y, 0)
        self.train_y = extract_polarization(self.train_y, 0)
        self.test_x = extract_polarization(self.test_x, 0)
        self.train_x = extract_polarization(self.train_x, 0)
        self.val_x = extract_polarization(self.val_x, 0)
        self.val_y = extract_polarization(self.val_y, 0)
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class HeraDeltaNormLoader(RawDataLoader):
    def load_data(self):
        file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all_delta_norm.pkl")
        data, masks = np.load(
            file_path, allow_pickle=True
        )
        data = np.expand_dims(data[:, 0, :, :], 1)
        masks = np.expand_dims(masks[:, 0, :, :], 1)
        train_x, train_y, test_x, test_y = test_train_split(data, masks)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.train_x, self.train_y, self.val_x, self.val_y = test_train_split(self.train_x,
                                                                              self.train_y)
        self.limit_datasets()
        self.original_size = self.train_x.shape[-1]
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()
