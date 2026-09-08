import argparse
import copy
import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from comparison import compare_models, compute_comparison_metrics
from dataset_folder.myDataset import MyDataset
import frameworks
from frameworks.framework import Framework

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

import models

model_list = {}

for _, module_name, _ in pkgutil.iter_modules(frameworks.__path__):
    importlib.import_module(f"frameworks.{module_name}")

for _, module_name, _ in pkgutil.iter_modules(models.__path__):
    try:
        module = importlib.import_module(f"models.{module_name}")
    except Exception:
        continue

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            model_list[name] = obj


def _resolve_config_path(config_path: str | Path | None) -> Path:
    project_root = Path(__file__).resolve().parent
    configs_dir = project_root / "configs"

    if config_path is None:
        target = configs_dir / "config.yaml"
    else:
        candidate = Path(config_path)
        if candidate.is_absolute():
            target = candidate
        elif candidate.parts and candidate.parts[0] == "configs":
            target = project_root / candidate
        else:
            target = configs_dir / candidate.name if candidate.suffix.lower() in {".yaml", ".yml"} else configs_dir / candidate

    resolved = target.resolve()
    try:
        resolved.relative_to(configs_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Config file must be inside the configs folder: {resolved}"
        ) from exc

    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found at {resolved}")

    return resolved


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_config_path(config_path)

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read YAML configuration files. "
            "Install it with `pip install pyyaml`."
        )

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a dictionary")

    return config


def _parse_model_params(raw_model_params: Any) -> dict[str, Any]:
    if raw_model_params is None:
        return {}

    if isinstance(raw_model_params, str):
        raw_model_params = json.loads(raw_model_params)

    if not isinstance(raw_model_params, dict):
        raise ValueError("Model parameters must be a JSON object or a YAML dictionary")

    return raw_model_params


def _merge_runtime_config(config: dict[str, Any] | None, cli_args: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config or {})
    runtime = {
        "model": cli_args.get("model"),
        "model_file": cli_args.get("model_file"),
        "technique": cli_args.get("technique"),
        "framework": cli_args.get("framework"),
        "model_params": cli_args.get("model_params"),
        "dataset_path": cli_args.get("dataset_path"),
        "dataset_files": cli_args.get("dataset_files"),
        "dataset_class": cli_args.get("dataset_class"),
    }

    for key, value in runtime.items():
        if value is None:
            continue
        if key == "model_params":
            yaml_model_params = dict(merged.get("model_params") or {})
            cli_model_params = _parse_model_params(value)
            yaml_model_params.update(cli_model_params)
            merged[key] = yaml_model_params
        elif key == "dataset_files" and isinstance(value, str):
            merged[key] = json.loads(value)
        else:
            merged[key] = value

    if "model_params" not in merged:
        merged["model_params"] = {}

    return merged


def _load_pretrained_model(model_path: Path, model_instance):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load pretrained checkpoint files") from exc

    try:
        ckpt = torch.load(model_path, map_location="cpu")

        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt)
            model_instance.load_state_dict(state_dict)
            return model_instance

        if isinstance(ckpt, torch.nn.Module):
            model_instance.load_state_dict(ckpt.state_dict())
            return model_instance

        raise RuntimeError(
            f"Unsupported checkpoint format {type(ckpt)}; expected state dict or nn.Module"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load pretrained model from {model_path}: {exc}") from exc


def _resolve_model_path(model_file: str, base_dir: Path | None = None) -> Path:
    candidates = [Path(model_file)]
    if base_dir is not None:
        candidates.append(base_dir / model_file)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Model file not found at {model_file}")


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        raise ValueError("Dataset path is required.")

    project_root = Path(__file__).resolve().parent
    candidate = Path(dataset_path)
    if candidate.is_absolute():
        target = candidate
    else:
        target = project_root / candidate

    resolved = target.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found at {resolved}")

    return resolved


def _load_dataset_from_path(dataset_path: str | Path | None, dataset_class: str | None = None, split: str = "test"):
    resolved_path = _resolve_dataset_path(dataset_path)

    if resolved_path.suffix.lower() in {".pkl", ".pickle"}:
        return MyDataset(source=resolved_path, split=split)

    if resolved_path.suffix.lower() != ".py":
        raise ValueError(f"Dataset path must point to a Python or pickle file: {resolved_path}")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("custom_dataset_module", resolved_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import dataset module from {resolved_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        candidate_names = []
        if dataset_class:
            candidate_names.append(dataset_class)
        candidate_names.extend(["dataset", "Dataset", "MyDataset", "load_dataset", "get_dataset"])

        for name in candidate_names:
            obj = getattr(module, name, None)
            if obj is None:
                continue

            if isinstance(obj, type):
                try:
                    return obj(split=split)
                except TypeError:
                    return obj()

            if callable(obj):
                try:
                    return obj(split=split)
                except TypeError:
                    return obj()

            return obj

        raise AttributeError(
            f"No usable dataset object found in {resolved_path}. "
            "Expected a dataset instance, a dataset class, or a load_dataset/get_dataset function."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from {resolved_path}: {exc}") from exc


def _load_dataset_files(dataset_files: Any, split: str = "test", dataset_class: str | None = None):
    """Load files that together form one dataset and select its split."""
    if isinstance(dataset_files, list):
        if not dataset_files:
            raise ValueError("dataset_files must be a non-empty list of paths")

        complete_datasets = [
            _load_dataset_from_path(path, dataset_class=dataset_class, split="all")
            for path in dataset_files
        ]
        complete_dataset = complete_datasets[0]
        if len(complete_datasets) > 1:
            complete_dataset = ConcatDataset(complete_datasets)
        return MyDataset(source=complete_dataset, split=split)

    if not isinstance(dataset_files, dict):
        raise ValueError("dataset_files must be a list of paths or a train/test mapping")

    paths = dataset_files.get(split)
    if paths is None:
        raise ValueError(f"dataset_files does not contain a '{split}' entry")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"dataset_files['{split}'] must be a path or a non-empty list of paths")

    datasets = [
        _load_dataset_from_path(path, dataset_class=dataset_class, split=split)
        for path in paths
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_test_loader(
    dataset: Any = None,
    batch_size: int = 32,
    shuffle: bool = False,
    feature_length: int | None = None,
):
    """Create a DataLoader for the selected dataset's test split.

    Works with:
    - a dataset object with a .test attribute
    - a dict like {'test': (X_test, y_test)}
    - a raw (features, labels) tuple
    - a custom dataset already returning (feature, label) or {'feature': ..., 'label': ...}
    """
    if dataset is None:
        raise ValueError("A dataset object is required to build the test loader.")

    if isinstance(dataset, MyDataset):
        adapted = dataset
        if feature_length is not None:
            adapted.feature_length = feature_length
        if getattr(adapted, "split", "train") != "test":
            adapted = MyDataset(source=dataset, split="test", feature_length=feature_length)
    elif hasattr(dataset, "test"):
        adapted = MyDataset(source=dataset.test, split="test", feature_length=feature_length)
    elif isinstance(dataset, dict) and "test" in dataset:
        adapted = MyDataset(source=dataset["test"], split="test", feature_length=feature_length)
    elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        adapted = MyDataset(
            features=dataset[0],
            labels=dataset[1],
            split="test",
            feature_length=feature_length,
        )
    elif hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        adapted = MyDataset(source=dataset, split="test", feature_length=feature_length)
    else:
        raise ValueError(
            "Unsupported dataset format. Provide a dataset, a .test split, or a (features, labels) tuple."
        )

    return DataLoader(
        adapted,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test compression frameworks and techniques on deep learning models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file containing the same parameters as the CLI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture to use"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to a pretrained model checkpoint file (.pth, .pt, .ckpt, .bin)"
    )
    parser.add_argument(
        "--technique",
        type=str,
        default=None,
        help="Compression technique to apply (e.g., pruning, quantization, distillation)"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Compression framework to use"
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default=None,
        help="JSON object with model initialization parameters"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to one dataset file; prefer dataset_files in YAML for train/test files"
    )
    parser.add_argument(
        "--dataset-files",
        type=str,
        default=None,
        help="JSON object with train/test dataset file paths"
    )
    parser.add_argument(
        "--dataset-class",
        type=str,
        default=None,
        help="Optional class name inside the dataset module to instantiate"
    )

    args = parser.parse_args()

    try:
        config_path = _resolve_config_path(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}")
        sys.exit(1)

    config = _load_yaml_config(config_path)
    runtime_config = _merge_runtime_config(config, vars(args))

    has_dataset_config = runtime_config.get("dataset_path") or runtime_config.get("dataset_files")
    missing = [key for key in ("model", "technique", "framework") if not runtime_config.get(key)]
    if not has_dataset_config:
        missing.append("dataset_path or dataset_files")
    if missing:
        parser.error(
            "Missing required arguments/config values for: " + ", ".join(missing)
        )

    model_name = runtime_config["model"]
    technique = runtime_config["technique"]
    framework_name = runtime_config["framework"]
    model_params = _parse_model_params(runtime_config.get("model_params", {}))

    print(f"Available models: {list(model_list.keys())}")
    model_class = model_list.get(model_name)

    if model_class is None:
        print(f"Error: Model '{model_name}' not found.")
        sys.exit(1)

    try:
        model = model_class(**model_params)
    except Exception as exc:
        print(f"Error instantiating model '{model_name}': {exc}")
        sys.exit(1)

    model_file = runtime_config.get("model_file")
    if model_file:
        try:
            resolved_path = _resolve_model_path(model_file, base_dir=Path(config_path).resolve().parent if config_path else None)
            if resolved_path.suffix.lower() not in {".pth", ".pt", ".ckpt", ".bin"}:
                raise ValueError("Only pretrained checkpoint files are supported")

            model = _load_pretrained_model(resolved_path, model)
        except Exception as exc:
            print(f"Error loading model: {exc}")
            sys.exit(1)

    framework_list = {
        cls.__name__: cls
        for cls in Framework.__subclasses__()
    }

    print(f"Available frameworks: {list(framework_list.keys())}")

    framework_class = framework_list.get(framework_name.lower().capitalize())
    if framework_class is None:
        print(
            f"Error: Framework '{framework_name}' not found in available frameworks: "
            f"{list(framework_list.keys())}"
        )
        sys.exit(1)

    framework_instance = framework_class(technique=technique)
    model_original = copy.deepcopy(model)
    model_compressed = framework_instance.compress(model, technique=technique)

    print(f"Model: {model_file}")
    print(f"Technique: {technique}")
    print(f"Framework: {framework_name}")
    print(f"Loaded model: {model_original}")
    print(f"Compressed model: {model_compressed}")

    dataset_path = runtime_config.get("dataset_path")
    dataset_files = runtime_config.get("dataset_files")
    dataset_class = runtime_config.get("dataset_class")

    try:
        if dataset_files:
            dataset = _load_dataset_files(dataset_files, split="test", dataset_class=dataset_class)
        else:
            dataset = _load_dataset_from_path(dataset_path, dataset_class=dataset_class, split="test")
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        sys.exit(1)

    test_loader = build_test_loader(
        dataset,
        batch_size=32,
        shuffle=False,
        feature_length=model_params.get("input_width"),
    )
  

    comparison_results = compare_models(
        model_original,
        model_compressed,
        test_loader,
        device="cpu"
    )
    print("Comparison results:")
    print(json.dumps(comparison_results, indent=4))

    final_results = compute_comparison_metrics(comparison_results, metric_name="speedup_factor")
    print("Final comparison metric (speedup_factor):", final_results)


if __name__ == "__main__":
    main()
