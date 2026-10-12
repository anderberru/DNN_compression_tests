from typing import Any

from .framework import Framework
from .calibrate_model import calibrate_model

class OpenVINONNCF(Framework):
    """Post-training optimization with OpenVINO NNCF."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        try:
            import torch
            import nncf
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO NNCF requires PyTorch and NNCF. "
                "Install them with `pip install nncf`."
            ) from exc

        if not isinstance(model, torch.nn.Module):
            raise TypeError("OpenVINO NNCF requires a torch.nn.Module model.")

        config = {**self.config, **kwargs}
        format_name = technique.casefold()

        # Quantization techniques and presets
        quantization_presets = {
            "int8": nncf.QuantizationPreset.PERFORMANCE,
            "performance": nncf.QuantizationPreset.PERFORMANCE,
            "int8_performance": nncf.QuantizationPreset.PERFORMANCE,
            "mixed": nncf.QuantizationPreset.MIXED,
            "int8_mixed": nncf.QuantizationPreset.MIXED,
        }

        # Pruning techniques
        pruning_formats = {
            "prune", "pruning", "magnitude", "magnitude_global", "magnitude_local"
        }

        # Quantization
        if format_name in quantization_presets:
            calibration_dataset = config.pop("calibration_dataset", None)
            dummy_input = config.pop("dummy_input", None)
            preset = config.pop("preset", None)
            subset_size = int(config.pop("subset_size", 300))
            fast_bias_correction = bool(config.pop("fast_bias_correction", True))

            if config:
                raise TypeError(
                    "Unsupported OpenVINO NNCF configuration options: "
                    f"{', '.join(sorted(config))}"
                )
            if subset_size < 1:
                raise ValueError("subset_size must be at least 1")

            if preset is None:
                preset = quantization_presets[format_name]
            elif isinstance(preset, str):
                try:
                    preset = quantization_presets[preset.casefold()]
                except KeyError as exc:
                    raise ValueError(
                        "preset must be 'performance' or 'mixed'"
                    ) from exc

            if calibration_dataset is None:
                if dummy_input is None:
                    raise ValueError(
                        "NNCF PTQ requires calibration_dataset or dummy_input."
                    )
                calibration_dataset = [dummy_input]
                subset_size = min(subset_size, 1)

            if not isinstance(calibration_dataset, nncf.Dataset):
                calibration_dataset = nncf.Dataset(calibration_dataset)

            return nncf.quantize(
                model,
                calibration_dataset,
                preset=preset,
                subset_size=subset_size,
                fast_bias_correction=fast_bias_correction,
            )

        # Pruning
        elif format_name in pruning_formats:
            try:
                from nncf.parameters import PruneMode
            except ImportError as exc:
                raise RuntimeError(
                    "NNCF pruning requires a recent version of NNCF. Update with `pip install -U nncf`."
                ) from exc

            # Arguments for pruning
            ratio = float(config.pop("ratio", config.pop("target_sparsity", 0.5)))
            examples_inputs = config.pop("examples_inputs", config.pop("dummy_input", None))
            ignored_scope = config.pop("ignored_scope", None)
            mode_str = config.pop("mode", format_name)

            if mode_str in {"magnitude_local", "local"}:
                prune_mode = PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL
            else:
                prune_mode = PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL

            if config:
                raise TypeError(
                    "Unsupported OpenVINO NNCF pruning configuration options: "
                    f"{', '.join(sorted(config))}"
                )

            # Apply pruning to the model
            return nncf.prune(
                model,
                mode=prune_mode,
                ratio=ratio,
                examples_inputs=examples_inputs,
                ignored_scope=ignored_scope
            )

        else:
            raise ValueError(
                f"Unsupported OpenVINO NNCF technique: {technique}"
            )
        