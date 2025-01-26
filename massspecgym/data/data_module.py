import typing as T
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional
from torch.utils.data.dataset import Subset

from torch.utils.data.dataloader import DataLoader
from massspecgym.data.datasets import MassSpecDataset, MSnDataset


class MassSpecDataModule(pl.LightningDataModule):
    """
    Data module containing a mass spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    """

    def __init__(
        self,
        dataset: MassSpecDataset,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = True,
        split_pth: Optional[Path] = None,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "identifier" and "fold",
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test"
                values. Default is None, in which case the split from the `dataset` is used.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Pre-processing to be executed only on a single main device when using distributed training."""
        pass
        # if self.split_pth is None:
        #     if isinstance(self.dataset, MSnDataset):
        #         # Filter metadata to only include root identifiers
        #         self.split = self.dataset.metadata[self.dataset.metadata["identifier"].str.endswith("_0000000")][
        #             ["identifier", "fold"]]
        #     else:
        #         self.split = self.dataset.metadata[["identifier", "fold"]]
        # else:
        #     # NOTE: custom split is not tested
        #     self.split = pd.read_csv(self.split_pth, sep="\t")
        #     if set(self.split.columns) != {"identifier", "fold"}:
        #         raise ValueError('Split file must contain "id" and "fold" columns.')
        #     self.split["identifier"] = self.split["identifier"].astype(str)
        #
        #     if isinstance(self.dataset, MSnDataset):
        #         # Use root identifiers from the dataset
        #         dataset_identifiers = set(self.dataset.root_identifier_to_index.keys())
        #     else:
        #         dataset_identifiers = set(self.dataset.metadata["identifier"])
        #
        #     split_identifiers = set(self.split["identifier"])
        #
        #     if dataset_identifiers != split_identifiers:
        #         print("Warning: Some identifiers in the split file are not found in the dataset. Taking intersection.")
        #         common_ids = dataset_identifiers.intersection(split_identifiers)
        #         # Filter the split to include only common identifiers
        #         self.split = self.split[self.split["identifier"].isin(common_ids)]
        #
        # self.split = self.split.set_index("identifier")["fold"]
        # if not set(self.split) <= {"train", "val", "test"}:
        #     raise ValueError(
        #         '"Folds" column must contain only "train", "val", or "test" values.'
        #     )

    def setup(self, stage=None):
        """Pre-processing to be executed on every device when using distributed training."""

        if self.split_pth is None:
            if isinstance(self.dataset, MSnDataset):
                # Filter metadata to only include root identifiers
                self.split = self.dataset.metadata[self.dataset.metadata["identifier"].str.endswith("_0000000")][
                    ["identifier", "fold"]]
            else:
                self.split = self.dataset.metadata[["identifier", "fold"]]
        else:
            # NOTE: custom split is not tested
            self.split = pd.read_csv(self.split_pth, sep="\t")
            if set(self.split.columns) != {"identifier", "fold"}:
                raise ValueError('Split file must contain "identifier" and "fold" columns.')
            self.split["identifier"] = self.split["identifier"].astype(str)

            if isinstance(self.dataset, MSnDataset):
                # Use root identifiers from the dataset
                dataset_identifiers = set(self.dataset.root_identifier_to_index.keys())
            else:
                dataset_identifiers = set(self.dataset.metadata["identifier"])

            split_identifiers = set(self.split["identifier"])

            if dataset_identifiers != split_identifiers:
                print("Warning: Some identifiers in the split file are not found in the dataset. Taking intersection.")
                common_ids = dataset_identifiers.intersection(split_identifiers)
                # Filter the split to include only common identifiers
                self.split = self.split[self.split["identifier"].isin(common_ids)]

        self.split = self.split.set_index("identifier")["fold"]
        if not set(self.split) <= {"train", "val", "test"}:
            raise ValueError(
                '"Folds" column must contain only "train", "val", or "test" values.'
            )

        if isinstance(self.dataset, MSnDataset):
            root_identifier_to_index = self.dataset.root_identifier_to_index

            fold_indices = {'train': [], 'val': [], 'test': []}

            for identifier, fold in self.split.items():
                index = root_identifier_to_index.get(identifier)
                if index is not None:
                    fold_indices[fold].append(index)
                else:
                    print(f"Warning: Identifier {identifier} not found in dataset.")
            if stage == "fit" or stage is None:
                self.train_dataset = Subset(self.dataset, fold_indices['train'])
                self.val_dataset = Subset(self.dataset, fold_indices['val'])
                print(f"Train dataset size: {len(self.train_dataset)}")
                print(f"Val dataset size: {len(self.val_dataset)}")
            if stage == "test":
                self.test_dataset = Subset(self.dataset, fold_indices['test'])
                print(f"Test dataset size: {len(self.test_dataset)}")
        else:

            split_mask = self.split.loc[self.dataset.metadata["identifier"]].values
            if stage == "fit" or stage is None:
                self.train_dataset = Subset(self.dataset, np.where(split_mask == "train")[0])
                self.val_dataset = Subset(self.dataset, np.where(split_mask == "val")[0])
                print(f"Train dataset size: {len(self.train_dataset)}")
                print(f"Val dataset size: {len(self.val_dataset)}")
            if stage == "test":
                self.test_dataset = Subset(self.dataset, np.where(split_mask == "test")[0])
                print(f"Test dataset size: {len(self.test_dataset)}")

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)
