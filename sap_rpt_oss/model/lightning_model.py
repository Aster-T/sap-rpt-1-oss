from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from sap_rpt_oss.constants import ModelSize
from sap_rpt_oss.data.ds import RPTParquetDataset, RPTTableDataset
from sap_rpt_oss.data.tokenizer import Tokenizer
from sap_rpt_oss.model.torch_model import RPT


class LightningModel(LightningModule):
    def __init__(
        self,
        model_size: Union[ModelSize, str] = ModelSize.base,
        checkpoint: Optional[Union[str, Path]] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        regression_type: str = "l2",
        classification_type: str = "cross-entropy",
        num_regression_bins: int = 16,
        drop_constant_columns: bool = True,
        max_num_columns: int = 500,
        random_seed: int = 42,
        is_valid: bool = True,
    ):
        super().__init__()
        self.model_size = self._normalize_model_size(model_size)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.regression_type = regression_type
        self.classification_type = classification_type
        self.num_regression_bins = num_regression_bins
        self.drop_constant_columns = drop_constant_columns
        self.max_num_columns = max_num_columns
        self.random_seed = random_seed
        self.is_valid = is_valid

        self.checkpoint = self._resolve_checkpoint(checkpoint)
        self._weights_loaded = False
        self._train_dataset = None

        self.model = RPT(
            model_size=self.model_size,
            regression_type=self.regression_type,
            classification_type=self.classification_type,
        )
        self.tokenizer = Tokenizer(
            regression_type=self.regression_type,
            classification_type=self.classification_type,
            random_seed=self.random_seed,
            num_regression_bins=self.num_regression_bins,
            is_valid=self.is_valid,
        )

    @staticmethod
    def _normalize_model_size(model_size: Union[ModelSize, str]) -> ModelSize:
        if isinstance(model_size, ModelSize):
            return model_size
        if model_size not in ModelSize.__members__:
            raise ValueError(
                f"{model_size} is not a valid model size: {list(ModelSize.__members__)}"
            )
        return ModelSize[model_size]

    @staticmethod
    def _resolve_checkpoint(
        checkpoint: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        if checkpoint is None:
            return None

        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.exists():
            return checkpoint_path.resolve()

        return Path(
            hf_hub_download(repo_id="SAP/sap-rpt-1-oss", filename=str(checkpoint))
        )

    def setup(self, stage: Optional[str] = None):
        del stage
        if self.checkpoint is not None and not self._weights_loaded:
            # Load on CPU first and let Lightning place the module on the training device.
            self.model.load_weights(self.checkpoint, device=torch.device("cpu"))
            self._weights_loaded = True

    def forward(
        self,
        data: dict[str, torch.Tensor],
        is_regression: bool,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(data, is_regression=is_regression, labels=labels)

    def build_train_dataset(
        self,
        table,
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        **dataset_kwargs,
    ) -> RPTTableDataset:
        return RPTTableDataset(
            table=table,
            fit_size=fit_size,
            is_regression=is_regression,
            tokenizer=self.tokenizer,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
            drop_constant_columns=self.drop_constant_columns,
            max_num_columns=self.max_num_columns,
            random_seed=self.random_seed,
            **dataset_kwargs,
        )

    def build_train_dataloader(
        self,
        table,
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        **dataset_kwargs,
    ) -> DataLoader:
        dataset = self.build_train_dataset(
            table=table,
            fit_size=fit_size,
            is_regression=is_regression,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
            **dataset_kwargs,
        )
        return DataLoader(dataset, batch_size=None, shuffle=False)

    def build_train_dataset_from_root(
        self,
        root_dir: Union[str, Path],
        fit_size: Optional[Union[int, float]],
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        regression_keyword: str = "regression",
        **dataset_kwargs,
    ) -> RPTParquetDataset:
        return RPTParquetDataset(
            root_dir=root_dir,
            fit_size=fit_size,
            tokenizer=self.tokenizer,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
            drop_constant_columns=self.drop_constant_columns,
            max_num_columns=self.max_num_columns,
            random_seed=self.random_seed,
            regression_keyword=regression_keyword,
            **dataset_kwargs,
        )

    def build_train_dataloader_from_root(
        self,
        root_dir: Union[str, Path],
        fit_size: Optional[Union[int, float]],
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        regression_keyword: str = "regression",
        **dataset_kwargs,
    ) -> DataLoader:
        dataset = self.build_train_dataset_from_root(
            root_dir=root_dir,
            fit_size=fit_size,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
            regression_keyword=regression_keyword,
            **dataset_kwargs,
        )
        return DataLoader(dataset, batch_size=None, shuffle=False)

    def set_training_data(
        self,
        table,
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        **dataset_kwargs,
    ):
        self._train_dataset = self.build_train_dataset(
            table=table,
            fit_size=fit_size,
            is_regression=is_regression,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
            **dataset_kwargs,
        )
        return self

    def set_training_data_from_root(
        self,
        root_dir: Union[str, Path],
        fit_size: Optional[Union[int, float]],
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        regression_keyword: str = "regression",
        **dataset_kwargs,
    ):
        self._train_dataset = self.build_train_dataset_from_root(
            root_dir=root_dir,
            fit_size=fit_size,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
            regression_keyword=regression_keyword,
            **dataset_kwargs,
        )
        return self

    def train_dataloader(self):
        if self._train_dataset is None:
            raise RuntimeError(
                "Training data is not configured. Call set_training_data(...) first "
                "or pass train_dataloaders=... to trainer.fit(...)."
            )
        return DataLoader(self._train_dataset, batch_size=None, shuffle=False)

    def training_step(self, batch, batch_idx):
        del batch_idx
        _, loss, metric = self(
            batch["data"],
            is_regression=bool(batch["is_regression"]),
            labels=batch["labels"],
        )

        metric_name = "train_r2" if batch["is_regression"] else "train_accuracy"
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(metric_name, metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.warmup_steps <= 0:
            return optimizer

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(float(step + 1) / float(self.warmup_steps), 1.0),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
