from typing import Any

from .framework import Framework


class Torchao(Framework):

    def __init__(self, technique: str, **kwargs):
        super().__init__(technique=technique, **kwargs)

    def compress(self, model, technique, **kwargs):

        from torchao.quantization import (
            quantize_,
            Int8DynamicActivationInt8WeightConfig,
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
        )

        if technique == "int8":
            config = Int8DynamicActivationInt8WeightConfig()
        elif technique == "float8":
            config = Float8DynamicActivationFloat8WeightConfig()
        elif technique == "float8_weight_only":
            config = Float8WeightOnlyConfig()
        else:
            raise ValueError(f"Unsupported compression technique: {technique}")

        quantize_(model, config)
        return model