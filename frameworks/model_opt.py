from typing import Any

from .framework import Framework


def calibrate_model(model: Any, dummy_input: Any, calibration_steps: int = 1) -> None:
    """Run calibration forward passes using the configured dummy input."""
    import torch

    if dummy_input is None:
        raise ValueError("dummy_input is required for calibration-based quantization")
    if calibration_steps < 1:
        raise ValueError("calibration_steps must be at least 1")

    def move_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [move_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {key: move_to_device(item, device) for key, item in value.items()}
        return value

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    was_training = model.training
    model.eval()
    inputs = move_to_device(dummy_input, device)
    with torch.no_grad():
        for _ in range(calibration_steps):
            if isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
    model.train(was_training)


class ModelOptimizer(Framework):
    """Model optimizer framework implementation that inherits Framework."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Compress a model using the requested technique."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )
        dummy_input = kwargs.pop("dummy_input", None)
        calibration_steps = kwargs.pop("calibration_steps", 1)
        import modelopt.torch.quantization as mtq

        forward_loop = lambda calibrated_model: calibrate_model(
            calibrated_model, dummy_input, calibration_steps
        )
        # if technique not in {"prune", "pruning"}:
        #     raise ValueError(
        #         "NVIDIA Model Optimizer only supports the 'pruning' technique"
        #     )

        if technique in {"prune", "pruning"}:
            from modelopt.torch.sparsity import sparsify, export

            config = {**self.config, **kwargs}
            config.pop("dummy_input", None)
            mode = config.pop("mode", "sparse_magnitude")
            search_config = config.pop("search_config", config or None)

            compressed_model = sparsify(
                model,
                mode=mode,
                config=search_config,
            )
            compressed_model = export(compressed_model)

        elif technique == "AWQ_int4":
            config = mtq.INT4_AWQ_CFG

            compressed_model = mtq.quantize(
                model,
                config,
                forward_loop=forward_loop,
            )
        elif technique == "SmoothQuant_int8":
            config = mtq.INT8_SMOOTHQUANT_CFG

            compressed_model = mtq.quantize(
                model,
                config,
                forward_loop=forward_loop,
            )
        elif technique == "FP8":
            config = mtq.FP8_DEFAULT_CFG

            compressed_model = mtq.quantize(
                model,
                config,
                forward_loop=forward_loop,
            )
        elif technique == "KV_Cache_FP8":
            config = mtq.FP8_KV_CFG

            compressed_model = mtq.quantize(
                model,
                config,
                forward_loop=forward_loop,
            )
        elif technique == "NVFP4": # cuda only
            config = mtq.NVFP4_DEFAULT_CFG 

            compressed_model = mtq.quantize(
                model,
                config,
                forward_loop=forward_loop,
            )
        else:
            raise ValueError(f"Unsupported compression technique: {technique}")
        
        return compressed_model
