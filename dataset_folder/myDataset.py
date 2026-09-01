from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Adapter for any imported dataset that should expose the format used later.

    It can wrap:
    - a raw (features, labels) pair
    - a dict with {'train': ..., 'test': ...}
    - a custom dataset object whose __getitem__ returns (feature, label) or a dict
    - a dataset object with a .test or .train attribute

    The adapted dataset always returns:
        {'feature': ..., 'label': ...}
    """

    def __init__(
        self,
        source: Any = None,
        features: Any = None,
        labels: Any = None,
        split: str = "train",
        data: dict[str, Any] | None = None,
        train_ratio: float = 0.8,
        feature_key: str = "feature",
        label_key: str = "label",
    ) -> None:
        self.split = str(split).lower()
        self.feature_key = feature_key
        self.label_key = label_key

        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.source = source
        self.features = None
        self.labels = None
        self._wrapped_dataset = None

        if source is not None:
            self._init_from_source(source)
        elif data is not None:
            self._init_from_data(data)
        elif features is not None or labels is not None:
            self._init_from_arrays(features, labels)
        else:
            raise ValueError("You must provide a source dataset, a (features, labels) pair, or a data dict.")

    def _init_from_source(self, source: Any) -> None:
        if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
            self._wrapped_dataset = source
            return

        if isinstance(source, dict):
            if self.split in source:
                self._init_from_source(source[self.split])
                return
            if "test" in source or "train" in source:
                self._init_from_source(source["test" if self.split == "test" else "train"])
                return

        if isinstance(source, (tuple, list)) and len(source) == 2:
            self._init_from_arrays(source[0], source[1])
            return

        raise TypeError("Unsupported source dataset format. Expected a dataset object, (features, labels), or a split dict.")

    def _init_from_data(self, data: dict[str, Any]) -> None:
        if "train" in data or "test" in data:
            subset = data["test" if self.split == "test" else "train"]
            self._init_from_source(subset)
            return

        if "features" in data and "labels" in data:
            self._init_from_arrays(data["features"], data["labels"])
            return

        if "feature" in data and "label" in data:
            self._init_from_arrays(data["feature"], data["label"])
            return

        raise ValueError("Unsupported data dict format. Use {'train': ..., 'test': ...} or {'features': ..., 'labels': ...}.")

    def _init_from_arrays(self, features: Any, labels: Any) -> None:
        self.features = torch.as_tensor(features)
        self.labels = torch.as_tensor(labels)

        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must contain the same number of samples")

        if self.split == "train":
            n_total = len(self.features)
            train_size = int(n_total * 0.8)
            self.features = self.features[:train_size]
            self.labels = self.labels[:train_size]
        else:
            n_total = len(self.features)
            train_size = int(n_total * 0.8)
            self.features = self.features[train_size:]
            self.labels = self.labels[train_size:]

    @staticmethod
    def _normalize_sample(sample: Any) -> dict[str, Any]:
        if isinstance(sample, dict):
            if "feature" in sample and "label" in sample:
                return {"feature": sample["feature"], "label": sample["label"]}
            if "input" in sample and "target" in sample:
                return {"feature": sample["input"], "label": sample["target"]}
            if "x" in sample and "y" in sample:
                return {"feature": sample["x"], "label": sample["y"]}
            if "data" in sample and "target" in sample:
                return {"feature": sample["data"], "label": sample["target"]}

        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            return {"feature": sample[0], "label": sample[1]}

        raise ValueError(
            "Unsupported sample format. Return (feature, label) or a dict with feature/label, input/target, or x/y."
        )

    def __len__(self) -> int:
        if self._wrapped_dataset is not None:
            return len(self._wrapped_dataset)
        if self.features is not None:
            return len(self.features)
        return 0

    def __getitem__(self, index: int):
        if self._wrapped_dataset is not None:
            sample = self._wrapped_dataset[index]
            return self._normalize_sample(sample)

        if self.features is not None and self.labels is not None:
            return {
                "feature": self.features[index],
                "label": self.labels[index],
            }

        raise IndexError("Dataset is empty")

    @staticmethod
    def _to_pair(sample: Any):
        normalized = MyDataset._normalize_sample(sample)
        return normalized["feature"], normalized["label"]
