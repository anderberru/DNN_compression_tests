from typing import Any

from .framework import Framework
from .calibrate_model import calibrate_model

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
