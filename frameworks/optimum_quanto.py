from copy import deepcopy
from typing import Any

from .framework import Framework
from .calibrate_model import calibrate_model

class OptimumQuanto(Framework):
    """Post-training quantization with Hugging Face Optimum Quanto."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Quantize and freeze a PyTorch model using Optimum Quanto."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        try:
            import torch
            from optimum.quanto import (
                Calibration,
                freeze,
                qfloat8,
                qint2,
                qint4,
                qint8,
                quantize,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Optimum Quanto quantization requires PyTorch and optimum-quanto. "
                "Install with `pip install optimum-quanto`."
            ) from exc

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Optimum Quanto requires a torch.nn.Module model.")

        config = {**self.config, **kwargs}
        weights = config.pop("weights", technique)
        activations = config.pop("activations", None)
        calibration_dataset = config.pop("calibration_dataset", None)
        dummy_input = config.pop("dummy_input", None)
        calibration_steps = int(config.pop("calibration_steps", 300))
        momentum = float(config.pop("momentum", 0.9))
        include = config.pop("include", None)
        exclude = config.pop("exclude", None)
        inplace = bool(config.pop("inplace", False))
        if config:
            raise TypeError(
                "Unsupported Optimum Quanto configuration options: "
                f"{', '.join(sorted(config))}"
            )
        if calibration_steps < 1:
            raise ValueError("calibration_steps must be at least 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the range [0.0, 1.0)")

        weight_types = {
            "int2": qint2,
            "qint2": qint2,
            "int4": qint4,
            "qint4": qint4,
            "int8": qint8,
            "qint8": qint8,
            "float8": qfloat8,
            "qfloat8": qfloat8,
        }
        if isinstance(weights, str):
            try:
                weights = weight_types[weights.casefold()]
            except KeyError as exc:
                raise ValueError(
                    "weights must be one of int2, int4, int8, or float8"
                ) from exc

        activation_types = {
            "int8": qint8,
            "qint8": qint8,
            "float8": qfloat8,
            "qfloat8": qfloat8,
        }
        if isinstance(activations, str):
            try:
                activations = activation_types[activations.casefold()]
            except KeyError as exc:
                raise ValueError(
                    "activations must be None, int8, or float8"
                ) from exc
        if activations is not None and calibration_dataset is None:
            if dummy_input is None:
                raise ValueError(
                    "Activation quantization requires calibration_dataset or dummy_input."
                )
            calibration_dataset = [dummy_input]
            calibration_steps = 1

        quantized_model = model if inplace else deepcopy(model)
        quantize(
            quantized_model,
            weights=weights,
            activations=activations,
            include=include,
            exclude=exclude,
        )

        if activations is not None:
            if isinstance(calibration_dataset, (torch.Tensor, dict, tuple)):
                calibration_dataset = [calibration_dataset]
            try:
                device = next(quantized_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            def move_to_device(value: Any) -> Any:
                if isinstance(value, torch.Tensor):
                    return value.to(device)
                if isinstance(value, tuple):
                    return tuple(move_to_device(item) for item in value)
                if isinstance(value, list):
                    return [move_to_device(item) for item in value]
                if isinstance(value, dict):
                    return {key: move_to_device(item) for key, item in value.items()}
                return value

            was_training = quantized_model.training
            quantized_model.eval()
            samples_calibrated = 0
            try:
                with Calibration(momentum=momentum), torch.no_grad():
                    for sample in calibration_dataset:
                        sample = move_to_device(sample)
                        if isinstance(sample, dict):
                            quantized_model(**sample)
                        elif isinstance(sample, (tuple, list)):
                            quantized_model(*sample)
                        else:
                            quantized_model(sample)
                        samples_calibrated += 1
                        if samples_calibrated >= calibration_steps:
                            break
            finally:
                quantized_model.train(was_training)
            if samples_calibrated == 0:
                raise ValueError("calibration_dataset yielded no samples")

        freeze(quantized_model)
        return quantized_model
