from typing import Any, Dict, Optional, Tuple, Callable

import torch
from lightning import LightningDataModule
import torch.utils
import torch.utils.data


class PINNDataModule(LightningDataModule):
    """`LightningDataModule` for the PINN dataset

    This Data Module will hold datasets that should be used for training and validation.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset=None,
        pred_dataset=None,
        batch_size: int = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:

        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset

        if isinstance(val_dataset, list):
            raise "Validation dataset cannot be a list."

        self.batch_size = batch_size
        self.data_train = None
        self.data_val = self.val_dataset

        self.data_test = self.test_dataset if self.test_dataset else self.val_dataset
        self.data_pred = self.pred_dataset if self.pre_dataset else self.val_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test()`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice!
        Also, it is called after `self.prepare_data)_` and there is a barrier in between which ensures that all processes
        proceed to `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to the setup. Either `"fit"`, `"validate"`, `"test"`, or `"pred"`
        """
        if self.data_train:
            self.data_train = torch.utils.data.DataLoader(
                dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the train dataloader

        :return: The train dataloader
        """
        return self.data_train

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the val dataloader

        :return: The val dataloader
        """
        return torch.utils.data.DataLoader(
            dataset=self.data_val, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the test dataloader

        :return: The test dataloader
        """
        return torch.utils.data.DataLoader(
            dataset=self.data_test, batch_size=self.batch_size, shuffle=False
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the pred dataloader

        :return: The pred dataloader
        """
        return torch.utils.data.DataLoader(
            dataset=self.data_pred, batch_size=self.batch_size, shuffle=False
        )

    def teardown(self, stage: str) -> None:
        """Lightning hook for clearning up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`,
        and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, `"predict"`.
        """
        return super().teardown(stage)
