from typing import Any

from .framework import Framework


class QKeras(Framework):
    """Post-training quantization for TensorFlow/Keras models with QKeras."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Convert a float Keras model to a QKeras model without retraining."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        if not callable(getattr(model, "to_json", None)) or not callable(
            getattr(model, "get_weights", None)
        ):
            raise TypeError(
                "QKeras PTQ requires a TensorFlow/Keras model. "
                "Convert PyTorch models to Keras before compression."
            )

        config = {**self.config, **kwargs}
        config.pop("dummy_input", None)
        weight_bits = int(config.pop("weight_bits", 8))
        activation_bits = int(config.pop("activation_bits", 8))
        quantizer_config = config.pop("quantizer_config", None)
        custom_objects = config.pop("custom_objects", None)
        transfer_weights = bool(config.pop("transfer_weights", True))
        if config:
            raise TypeError(
                f"Unsupported QKeras configuration options: {', '.join(sorted(config))}"
            )
        if weight_bits < 1 or activation_bits < 1:
            raise ValueError("weight_bits and activation_bits must be positive")

        try:
            from qkeras.utils import model_quantize
        except ImportError as exc:
            raise RuntimeError(
                "QKeras PTQ requires compatible TensorFlow and QKeras installations. "
                "Install QKeras with `pip install qkeras`."
            ) from exc

        if quantizer_config is None:
            weight_quantizer = f"quantized_bits({weight_bits},0,1)"
            quantizer_config = {
                layer_name: {
                    "kernel_quantizer": weight_quantizer,
                    "bias_quantizer": weight_quantizer,
                }
                for layer_name in (
                    "QDense",
                    "QConv1D",
                    "QConv2D",
                    "QConv2DTranspose",
                )
            }
            quantizer_config.update(
                {
                    "QSeparableConv1D": {
                        "depthwise_quantizer": weight_quantizer,
                        "pointwise_quantizer": weight_quantizer,
                        "bias_quantizer": weight_quantizer,
                    },
                    "QSeparableConv2D": {
                        "depthwise_quantizer": weight_quantizer,
                        "pointwise_quantizer": weight_quantizer,
                        "bias_quantizer": weight_quantizer,
                    },
                    "QDepthwiseConv2D": {
                        "depthwise_quantizer": weight_quantizer,
                        "bias_quantizer": weight_quantizer,
                    },
                    "QLSTM": {
                        "kernel_quantizer": weight_quantizer,
                        "recurrent_quantizer": weight_quantizer,
                        "bias_quantizer": weight_quantizer,
                    },
                    "QGRU": {
                        "kernel_quantizer": weight_quantizer,
                        "recurrent_quantizer": weight_quantizer,
                        "bias_quantizer": weight_quantizer,
                    },
                    "QSimpleRNN": {
                        "kernel_quantizer": weight_quantizer,
                        "recurrent_quantizer": weight_quantizer,
                        "bias_quantizer": weight_quantizer,
                    },
                    "QActivation": {
                        "relu": f"quantized_relu({activation_bits})",
                        "tanh": f"quantized_tanh({activation_bits})",
                        "sigmoid": f"quantized_sigmoid({activation_bits})",
                    },
                }
            )

        return model_quantize(
            model,
            quantizer_config,
            activation_bits=activation_bits,
            custom_objects=custom_objects,
            transfer_weights=transfer_weights,
        )
