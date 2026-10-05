from typing import Any

from .framework import Framework


class IntelNeuralCompressor(Framework):
    """Post-training PyTorch quantization with Intel Neural Compressor."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Apply INC round-to-nearest weight-only post-training quantization."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        try:
            import torch
            from neural_compressor.torch.quantization import RTNConfig, quantize
        except ImportError as exc:
            raise RuntimeError(
                "Intel Neural Compressor PyTorch quantization requires PyTorch and "
                "Intel Neural Compressor. Install with `pip install "
                "neural-compressor[torch]`."
            ) from exc

        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "Intel Neural Compressor PTQ requires a torch.nn.Module model."
            )

        config = {**self.config, **kwargs}
        dummy_input = config.pop("dummy_input", None)
        quant_config = config.pop("quant_config", None)
        inplace = bool(config.pop("inplace", False))

        if quant_config is None:
            rtn_formats = {
                "rtn": ("int", 8),
                "int": ("int", 8),
                "int8": ("int8", 8),
                "int4": ("int4", 4),
                "nf4": ("nf4", 4),
                "fp4": ("fp4", 4),
                "fp4_e2m1": ("fp4_e2m1", 4),
                "fp4_e2m1_bnb": ("fp4_e2m1_bnb", 4),
                "fp8": ("fp8_e4m3fn", 8),
                "float8": ("fp8_e4m3fn", 8),
                "float8_weight_only": ("fp8_e4m3fn", 8),
                "fp8_e4m3fn": ("fp8_e4m3fn", 8),
                "fp8_e4m3fnuz": ("fp8_e4m3fnuz", 8),
                "fp8_e5m2": ("fp8_e5m2", 8),
                "fp8_e5m2fnuz": ("fp8_e5m2fnuz", 8),
            }
            format_name = technique.casefold()
            if format_name not in rtn_formats:
                raise ValueError(
                    f"Unsupported Intel Neural Compressor quantization technique: {technique}"
                )
            default_dtype, default_bits = rtn_formats[format_name]
            dtype = config.pop("dtype", default_dtype)
            bits = config.pop("bits", None)
            weight_bits = config.pop("weight_bits", None)
            if bits is not None and weight_bits is not None:
                raise ValueError("Specify only one of bits or weight_bits")
            if bits is None:
                bits = weight_bits if weight_bits is not None else default_bits
            bits = int(bits)
            group_size = int(config.pop("group_size", 32))
            use_sym = bool(config.pop("use_sym", True))
            if bits < 1:
                raise ValueError("bits must be a positive integer")
            if group_size == 0 or group_size < -1:
                raise ValueError("group_size must be -1 or a positive integer")
            quant_config = RTNConfig(
                dtype=dtype,
                bits=bits,
                group_size=group_size,
                use_sym=use_sym,
            )

        if config:
            raise TypeError(
                "Unsupported Intel Neural Compressor configuration options: "
                f"{', '.join(sorted(config))}"
            )

        return quantize(
            model,
            quant_config,
            inplace=inplace,
            example_inputs=dummy_input,
        )
