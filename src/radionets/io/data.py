from pathlib import Path

import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


class H5DataSet(Dataset):
    def __init__(self, data_dir, tar_fourier, mode="train"):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        bundle_paths = data_dir.glob(f"samp_{mode}_*.h5")
        self.bundles = natsorted(bundle_paths)
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))
        self.tar_fourier = tar_fourier

        if not self.bundles:
            raise ValueError("No bundles found! Please check the names of your files.")

    def __call__(self):
        return print("This is the H5DataSet class.")

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        x = self.open_image("x", i)
        y = self.open_image("y", i)
        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def open_image(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])

        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)

        indices, _ = torch.sort(i)
        bundle = torch.div(indices, self.num_img, rounding_mode="floor")
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)

        bundle_paths = [
            h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
        ]
        bundle_paths_str = list(map(str, bundle_paths))

        data = torch.from_numpy(
            np.array(
                [
                    bund[var][img]
                    for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                    for img in image[
                        bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                    ]
                ]
            )
        )

        if self.tar_fourier is False and data.shape[1] == 2:
            raise ValueError(
                "Two channeled data is used despite Fourier being False.\
                    Set Fourier to True!"
            )

        if data.shape[0] == 1:
            data = data.squeeze(0)
        return data.float()
