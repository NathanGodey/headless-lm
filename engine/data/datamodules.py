import pytorch_lightning as pl
from typing import Optional
from engine.data.datasets import NlpDataSource
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    _default_train_batch_size = 1
    _default_infer_batch_size = 1

    def __init__(self, train_batch_size: int = 1, infer_batch_size: int = 1,
                 split_names=None, from_disk=False, num_workers=1):
        super().__init__()
        self.datasource = NlpDataSource(split_names=split_names, from_disk=from_disk)

        self.trainset = None
        self.valset = None
        self.num_workers=num_workers

        self.name = 'NA'

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size

    @classmethod
    def from_datasets(cls, *args, **kwargs):
        module = cls(
            train_batch_size=kwargs.pop('train_batch_size', cls._default_train_batch_size),
            infer_batch_size=kwargs.pop('infer_batch_size', cls._default_infer_batch_size),
            split_names=kwargs.pop('split_names', None),
            from_disk=kwargs.pop('from_disk', False),
            num_workers=kwargs.pop('num_workers', 1)
            )
        if args:
            module.name = args[0]
        module.datasource.load(*args, **kwargs)
        return module

    def filter(self, new_filter, **kwargs):
        self.datasource.filter(new_filter, **kwargs)

    def map(self, new_map, **kwargs):
        self.datasource.map(new_map, **kwargs)

    def shuffle(self, **kwargs):
        self.datasource.shuffle(**kwargs)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.trainset = self.datasource.trainset
            self.valset = self.datasource.valset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.infer_batch_size, num_workers=self.num_workers)
