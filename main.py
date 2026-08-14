import argparse
import copy
import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any

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
    }

    for key, value in runtime.items():
        if value is None:
            continue
        if key == "model_params":
            yaml_model_params = dict(merged.get("model_params") or {})
            cli_model_params = _parse_model_params(value)
            yaml_model_params.update(cli_model_params)
            merged[key] = yaml_model_params
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

    args = parser.parse_args()

    try:
        config_path = _resolve_config_path(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}")
        sys.exit(1)

    config = _load_yaml_config(config_path)
    runtime_config = _merge_runtime_config(config, vars(args))

    missing = [key for key in ("model", "technique", "framework") if not runtime_config.get(key)]
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


if __name__ == "__main__":
    main()
