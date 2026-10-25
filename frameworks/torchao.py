from typing import Any


from .framework import Framework
from .calibrate_model import calibrate_model


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
        from torchao.sparsity import sparsify_, semi_sparse_weight
        try:
            from torchao.sparsity import WandaSparsifier
        except ImportError:
            WandaSparsifier = None

        if technique == "int8":
            config = Int8DynamicActivationInt8WeightConfig()
            quantize_(model, config)
        elif technique == "float8":
            config = Float8DynamicActivationFloat8WeightConfig()
            quantize_(model, config)
        elif technique == "float8_weight_only":
            config = Float8WeightOnlyConfig()
            quantize_(model, config)

        elif technique in {"prune", "pruning", "semi_sparse", "2:4_sparsity"}:
            # Applies semi-structured sparsity (2:4)
            config = semi_sparse_weight()
            sparsify_(model, config)
            
        elif technique == "wanda": # Weight and Activation sparsity
            if WandaSparsifier is None:
                raise ImportError("WandaSparsifier is not available in your version of torchao.")
                
            sparsity_level = kwargs.pop("sparsity_level", 0.5)
            block_size = kwargs.pop("semi_structured_block_size", None)
            dummy_input = kwargs.pop("dummy_input", None)
            calibration_steps = kwargs.pop("calibration_steps", 1)

            import inspect
            num_params = len(inspect.signature(model.forward).parameters)
            
            if isinstance(dummy_input, (tuple, list)):
                dummy_input = tuple(dummy_input[:num_params])
            
            sparsifier = WandaSparsifier(
                sparsity_level=sparsity_level,
                semi_structured_block_size=block_size
            )
            
            # Change the model structure to add masks and observation hooks
            sparsifier.prepare(model, config=None)
            
            # Calibrate the model using dummy input
            if dummy_input is None:
                raise ValueError("Dummy input is required to calibrate the Wanda method.")
            calibrate_model(model, dummy_input, calibration_steps)
            
            # Decide what weights to zero out based on the calibration data
            sparsifier.step()
            
            # Apply the mask to the model weights and remove the mask parameters
            sparsifier.squash_mask()
        else:
            raise ValueError(f"Unsupported compression technique: {technique}")

        return model