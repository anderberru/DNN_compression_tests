from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
        feature_length: int | None = None,
        sequence_length: int | None = None,
    ) -> None:
        self.split = str(split).lower()
        self.feature_key = feature_key
        self.label_key = label_key
        self.train_ratio = train_ratio
        self.feature_length = feature_length
        self.sequence_length = sequence_length

        if sequence_length is not None and sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        if self.split not in {"train", "test", "all"}:
            raise ValueError("split must be 'train', 'test', or 'all'")

        self.source = source
        self.features = None
        self.labels = None
        self._wrapped_dataset = None
        self._indices = None

        if source is not None:
            if isinstance(source, (str, Path)):
                source = self._load_pickle(source)
            self._init_from_source(source)
        elif data is not None:
            self._init_from_data(data)
        elif features is not None or labels is not None:
            self._init_from_arrays(features, labels)
        else:
            raise ValueError("You must provide a source dataset, a (features, labels) pair, or a data dict.")

        if self._wrapped_dataset is not None and self.split in {"train", "test"}:
            split_index = int(len(self._wrapped_dataset) * self.train_ratio)
            if self.split == "train":
                self._indices = range(0, split_index)
            else:
                self._indices = range(split_index, len(self._wrapped_dataset))

    def _init_from_source(self, source: Any) -> None:
        if isinstance(source, dict):
            split_source = source.get(self.split)
            if split_source is not None:
                self._init_from_source(split_source)
                return

            split_key = f"{self.split}_data"
            if split_key in source:
                self._init_from_source(source[split_key])
                return

            if "features" in source and "labels" in source:
                self._init_from_arrays(source["features"], source["labels"])
                return

            if "feature" in source and "label" in source:
                self._init_from_arrays(source["feature"], source["label"])
                return

            if "cycle_data" in source:
                self._init_from_source(source["cycle_data"])
                return

            # BatteryML pickles may store samples in a mapping keyed by an
            # identifier instead of using zero-based integer indices.
            if source:
                self._wrapped_dataset = list(source.values())
                return

        if isinstance(source, (tuple, list)) and len(source) == 2:
            self._init_from_arrays(source[0], source[1])
            return

        if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
            self._wrapped_dataset = source
            return

        if isinstance(source, dict):
            if "test" in source or "train" in source:
                self._init_from_source(source["test" if self.split == "test" else "train"])
                return

        raise TypeError("Unsupported source dataset format. Expected a dataset object, (features, labels), or a split dict.")

    @staticmethod
    def _load_pickle(path: str | Path) -> Any:
        path = Path(path)
        if path.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(f"Unsupported dataset file type: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Pickle is executable during loading; only load trusted files.
        with path.open("rb") as handle:
            return pickle.load(handle)

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
            train_size = int(n_total * self.train_ratio)
            self.features = self.features[:train_size]
            self.labels = self.labels[:train_size]
        elif self.split == "test":
            n_total = len(self.features)
            train_size = int(n_total * self.train_ratio)
            self.features = self.features[train_size:]
            self.labels = self.labels[train_size:]

    def _normalize_sample(self, sample: Any) -> dict[str, Any]:
        if isinstance(sample, dict):
            if "feature" in sample and "label" in sample:
                return {
                    "feature": self._fit_feature_length(sample["feature"]),
                    "label": sample["label"],
                }
            if "input" in sample and "target" in sample:
                return {
                    "feature": self._fit_feature_length(sample["input"]),
                    "label": sample["target"],
                }
            if "x" in sample and "y" in sample:
                return {
                    "feature": self._fit_feature_length(sample["x"]),
                    "label": sample["y"],
                }
            if "data" in sample and "target" in sample:
                return {
                    "feature": self._fit_feature_length(sample["data"]),
                    "label": sample["target"],
                }
            if "features" in sample and "labels" in sample:
                return {
                    "feature": self._fit_feature_length(sample["features"]),
                    "label": sample["labels"],
                }
            if "voltage_in_V" in sample and "discharge_capacity_in_Ah" in sample:
                voltage = sample["voltage_in_V"]
                capacity = sample["discharge_capacity_in_Ah"]
                if not voltage or not capacity:
                    raise ValueError("BatteryML cycle has no voltage or discharge-capacity values")
                return {
                    "feature": self._fit_feature_length(
                        torch.as_tensor(voltage, dtype=torch.float32).unsqueeze(0)
                    ),
                    "label": torch.as_tensor(capacity[-1], dtype=torch.float32),
                }

        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            return {
                "feature": self._fit_feature_length(sample[0]),
                "label": sample[1],
            }

        raise ValueError(
            "Unsupported sample format. Return (feature, label) or a dict with feature/label, input/target, or x/y."
        )
    
    def __len__(self) -> int:
        if self._wrapped_dataset is not None:
            if self._indices is not None:
                length = len(self._indices)
            else:
                length = len(self._wrapped_dataset)
            if self.sequence_length is not None:
                return max(0, length - self.sequence_length + 1)
            return length
        if self.features is not None:
            length = len(self.features)
            if self.sequence_length is not None:
                return max(0, length - self.sequence_length + 1)
            return length
        return 0

    def __getitem__(self, index: int):
        if self._wrapped_dataset is not None:
            if self._indices is not None:
                if self.sequence_length is None:
                    index = self._indices[index]
                else:
                    end = index + self.sequence_length
                    indices = self._indices[index:end]
                    samples = [
                        self._normalize_sample(self._wrapped_dataset[item])
                        for item in indices
                    ]
                    features = torch.stack([item["feature"] for item in samples])
                    if features.ndim == 3 and features.shape[1] == 1:
                        features = features.squeeze(1)
                    return {
                        "feature": features,
                        "label": samples[-1]["label"],
                    }
            elif self.sequence_length is not None:
                samples = [
                    self._normalize_sample(self._wrapped_dataset[item])
                    for item in range(index, index + self.sequence_length)
                ]
                features = torch.stack([item["feature"] for item in samples])
                if features.ndim == 3 and features.shape[1] == 1:
                    features = features.squeeze(1)
                return {
                    "feature": features,
                    "label": samples[-1]["label"],
                }
            sample = self._wrapped_dataset[index]
            return self._normalize_sample(sample)

        if self.features is not None and self.labels is not None:
            if self.sequence_length is not None:
                end = index + self.sequence_length
                features = [
                    self._fit_feature_length(feature)
                    for feature in self.features[index:end]
                ]
                features = torch.stack(features)
                if features.ndim == 3 and features.shape[1] == 1:
                    features = features.squeeze(1)
                return {
                    "feature": features,
                    "label": self.labels[end - 1],
                }
            return {
                "feature": self._fit_feature_length(self.features[index]),
                "label": self.labels[index],
            }

        raise IndexError("Dataset is empty")

    def _fit_feature_length(self, feature: Any):
        feature = torch.as_tensor(feature, dtype=torch.float32)
        if self.feature_length is None or feature.shape[-1] == self.feature_length:
            return feature
        if feature.ndim != 2:
            raise ValueError(
                "feature_length can only resize one-dimensional features with shape [channels, length]"
            )
        return F.interpolate(
            feature.unsqueeze(0),
            size=self.feature_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    def _to_pair(self, sample: Any):
        normalized = self._normalize_sample(sample)
        return normalized["feature"], normalized["label"]
