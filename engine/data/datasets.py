from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk, DatasetDict, IterableDatasetDict, Dataset
import torch
import re


class DataFilters(list):
    def __call__(self, data):
        for filter in self:
            if not filter(data):
                return True
        return False


class DataMaps(list):
    def __call__(self, data):
        mapped_data = None
        for _map in self:
            mapped_data = _map(mapped_data) if mapped_data else _map(data)
        return mapped_data if mapped_data is not None else data


class NlpDataSource:

    def __init__(self, split_names=None, from_disk=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filters = DataFilters()
        self._maps = DataMaps()
        self.data = None
        self.split_names = split_names if split_names else ['train', 'validation', 'test']
        self.streamed = False
        self.from_disk=from_disk

    def _process_split(self, splits):
        decomposed_splits = []
        for split in splits:
            if '(' in split:
                split_name, percent_bot, _, percent_top, _ = re.findall('(.*)\((([0-9]*[.])?[0-9]+|)?:(([0-9]*[.])?[0-9]+|)?\)', split)[0]
                percent_bot = float(percent_bot) if percent_bot else 0
                percent_top = float(percent_top) if percent_top else 1
                decomposed_splits.append((split_name, percent_bot, percent_top))
            else:
                decomposed_splits.append((split, 0, 1))
        return decomposed_splits

    def load(self, *args, streamed=False, **kwargs):
        self.streamed = streamed
        self.split_names = kwargs.get('split_names') or self.split_names
        if self.from_disk:
            decomposed_splits = self._process_split(self.split_names)
            whole_ds = load_from_disk(*args, **kwargs)
            if isinstance(whole_ds, Dataset):
                whole_ds = DatasetDict({"train": whole_ds})
            self.data = DatasetDict({
                split_name: whole_ds[split].select(range(int(percent_bot*len(whole_ds[split])),int(percent_top*len(whole_ds[split]))))
                for split_name, (split, percent_bot, percent_top) in zip(self.split_names, decomposed_splits)
            })
        else:
            if streamed:
                self.data = IterableDatasetDict({
                    split_name: load_dataset(*args, streaming=streamed, split=split_name, **kwargs)
                    for split_name in self.split_names
                    })
            else:
                self.data = DatasetDict({
                    split_name: load_dataset(*args, streaming=streamed, split=split_name, **kwargs)
                    for split_name in self.split_names
                    })

    def filter(self, new_filter, **kwargs):
        if not self.streamed:
            self.data = self.data.filter(new_filter, **kwargs)
        self._filters.append(new_filter)

    def map(self, new_map, **kwargs):
        if not self.streamed:
            self.data = self.data.map(new_map, **kwargs)
        self._maps.append(new_map)
    
    def shuffle(self, **kwargs):
        if not self.streamed:
            self.data = self.data.shuffle(**kwargs)
    
    def column_names(self):
        return list(self.data.values())[0].column_names

    def _load_split(self, split_name):
        if self.streamed:
            return NlpIterableDataset(source=self, split_name=split_name)
        else:
            return self.data[split_name] if self.data else None

    @property
    def trainset(self):
        return self._load_split(self.split_names[0])

    @property
    def valset(self):
        return self._load_split(self.split_names[1])


class NlpIterableDataset(IterableDataset):

    def __init__(self, source=None, split_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filters = source._filters if source else []
        self._maps = source._maps if source else []

        self.data = source.data[split_name] if split_name else None

        self._data_iterator = iter(self.data) if self.data else None
        self._data_len = self.data.info.splits[split_name].num_examples if self.data else 0
        self.filtered_len = torch.inf
        self._running_filtered_len = self._data_len
        self._has_seen_data = False

    def __len__(self):
        return self.filtered_len

    def filter(self, new_filter):
        self._filters.append(new_filter)

    def _safe_iter(self):
        try:
            return next(self._data_iterator)
        except StopIteration as e:
            self._has_seen_data = True
            self._data_iterator = iter(self.data)
            self.filtered_len = self._running_filtered_len
            raise e

    def __iter__(self):
        return self

    def __next__(self):
        elem = self._safe_iter()
        while elem is not None and self._filters(elem):
            elem = self._safe_iter()
            if not self._has_seen_data:
                self._running_filtered_len -= 1
        return self._maps(elem)
