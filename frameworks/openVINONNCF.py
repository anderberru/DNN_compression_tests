from typing import Any

from .framework import Framework


class OpenVINONNCF(Framework):
    """Post-training INT8 quantization with OpenVINO NNCF."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Quantize a PyTorch model using NNCF and calibration samples."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        try:
            import torch
            import nncf
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO NNCF PTQ requires PyTorch and NNCF. "
                "Install them with `pip install nncf`."
            ) from exc

        if not isinstance(model, torch.nn.Module):
            raise TypeError("OpenVINO NNCF PTQ requires a torch.nn.Module model.")

        config = {**self.config, **kwargs}
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

        technique_presets = {
            "int8": nncf.QuantizationPreset.PERFORMANCE,
            "performance": nncf.QuantizationPreset.PERFORMANCE,
            "int8_performance": nncf.QuantizationPreset.PERFORMANCE,
            "mixed": nncf.QuantizationPreset.MIXED,
            "int8_mixed": nncf.QuantizationPreset.MIXED,
        }
        if preset is None:
            try:
                preset = technique_presets[technique.casefold()]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported OpenVINO NNCF quantization technique: {technique}"
                ) from exc
        elif isinstance(preset, str):
            try:
                preset = technique_presets[preset.casefold()]
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
