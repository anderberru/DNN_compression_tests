import argparse
import json
import sys
from pathlib import Path
from frameworks.framework import Framework

import pkgutil
import importlib
import frameworks

for _, module_name, _ in pkgutil.iter_modules(frameworks.__path__):
    importlib.import_module(f"frameworks.{module_name}")

import models
import inspect

model_list = {}

for _, module_name, _ in pkgutil.iter_modules(models.__path__):
    module = importlib.import_module(f"models.{module_name}")
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            model_list[name] = obj

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


def _resolve_model_path(model_file: str) -> Path:
    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model_path.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Test compression frameworks and techniques on deep learning models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model architecture to use"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=False,
        help="Path to a pretrained model checkpoint file (.pth, .pt, .ckpt, .bin)"
    )
    parser.add_argument(
        "--technique",
        type=str,
        required=True,
        help="Compression technique to apply (e.g., pruning, quantization, distillation)"
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        help="Compression framework to use"
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help="JSON object with model initialization parameters"
    )

    args = parser.parse_args()

    print(f"Available models: {list(model_list.keys())}")
    model_class = model_list.get(args.model)

    if model_class is None:
        print(
            f"Error: Model '{args.model}' not found. "
        )
        sys.exit(1)

    try:
        model_params = json.loads(args.model_params)
        if not isinstance(model_params, dict):
            raise ValueError("Model parameters must be a JSON object")
    except Exception as exc:
        print(f"Error parsing model parameters: {exc}")
        sys.exit(1)

    try:
        model = model_class(**model_params)
    except Exception as exc:
        print(f"Error instantiating model '{args.model}': {exc}")
        sys.exit(1)

    if args.model_file:
        try:
            resolved_path = _resolve_model_path(args.model_file)
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
    
    framework_class = framework_list.get(args.framework.lower().capitalize())
    if framework_class is None:
        print(f"Error: Framework '{args.framework}' not found in available frameworks: {list(framework_list.keys())}")
        sys.exit(1)
    framework_instance = framework_class(technique=args.technique)
    model_compressed = framework_instance.compress(model, technique=args.technique)

    
    

    print(f"Model: {args.model_file}")
    print(f"Technique: {args.technique}")
    print(f"Framework: {args.framework}")
    print(f"Loaded model: {model}")
    print(f"Compressed model: {model_compressed}")

if __name__ == "__main__":
    main()
