from copy import deepcopy
from typing import Any

from .framework import Framework
from .calibrate_model import calibrate_model

class MicrosoftNNI(Framework):
    """Post-training compression with Microsoft NNI."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Compress a PyTorch model using NNI's PTQ quantizer or Pruners."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        try:
            import torch
            import nni
        except ImportError as exc:
            raise RuntimeError(
                "Microsoft NNI requires PyTorch and NNI with compression "
                "support. Install with `pip install nni`."
            ) from exc

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Microsoft NNI requires a torch.nn.Module model.")

        config = {**self.config, **kwargs}
        technique_name = technique.casefold()
        inplace = bool(config.pop("inplace", False))

        pruning_formats = {
            "prune", "pruning", "level", "magnitude", 
            "l1", "l1norm", "l2", "l2norm", "fpgm"
        }

        # Pruning
        if technique_name in pruning_formats:
            try:
                import nni.compression.pruning as nni_pruning
            except ImportError as exc:
                raise RuntimeError("NNI pruning modules could not be loaded.") from exc

            config_list = config.pop("config_list", None)
            target_sparsity = float(config.pop("target_sparsity", 0.5))

            config.pop("dummy_input", None)
            config.pop("calibration_dataset", None)
            config.pop("calibration_steps", None)

            if config_list is None:
                config_list = [{
                    "op_types": ["Linear", "Conv2d"],
                    "sparse_ratio": target_sparsity
                }]
            elif not isinstance(config_list, list):
                raise TypeError("config_list must be a list of NNI pruning configs")

            if config:
                raise TypeError(
                    "Unsupported Microsoft NNI pruning configuration options: "
                    f"{', '.join(sorted(config))}"
                )

            pruned_model = model if inplace else deepcopy(model)

            # Pruning technique selection
            if technique_name in {"prune", "pruning", "level", "magnitude"}:
                pruner = nni_pruning.LevelPruner(pruned_model, config_list)
            elif technique_name in {"l1", "l1norm"}:
                pruner = nni_pruning.L1NormPruner(pruned_model, config_list)
            elif technique_name in {"l2", "l2norm"}:
                pruner = nni_pruning.L2NormPruner(pruned_model, config_list)
            elif technique_name == "fpgm":
                pruner = nni_pruning.FPGMPruner(pruned_model, config_list)
            else:
                raise ValueError(f"Unsupported NNI pruning technique: {technique}")

            # Execute compression and consolidate the model (removes NNI wrappers)
            _, masks = pruner.compress()
            pruner.unwrap_model()
            
            
            # Permanently inject zeros into the PyTorch model weights
            with torch.no_grad():
                for name, module in pruned_model.named_modules():
                    # NNI stores the mask with the module name
                    if name in masks and 'weight' in masks[name]:
                        # We multiply the original weight by the mask (0s and 1s)
                        mask_tensor = masks[name]['weight'].to(module.weight.device)
                        module.weight.mul_(mask_tensor)
            
            return pruned_model

        # Quantization (PTQ)
        try:
            import nni.compression.quantization as nni_quantization
            from nni.compression.utils import TorchEvaluator
        except ImportError as exc:
            raise RuntimeError("NNI quantization modules could not be loaded.") from exc

        calibration_dataset = config.pop("calibration_dataset", None)
        dummy_input = config.pop("dummy_input", None)
        config_list = config.pop("config_list", None)
        
        if technique_name in {"ptq", "int", "int8"}:
            technique_dtype = "int8"
        elif technique_name in {"uint", "uint8"}:
            technique_dtype = "uint8"
        elif technique_name.startswith(("int", "uint")):
            prefix = "uint" if technique_name.startswith("uint") else "int"
            try:
                technique_bits = int(technique_name.removeprefix(prefix))
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported Microsoft NNI quantization technique: {technique}"
                ) from exc
            if not 1 <= technique_bits <= 32:
                raise ValueError("Technique bit width must be between 1 and 32")
            technique_dtype = f"{prefix}{technique_bits}"
        else:
            raise ValueError(
                f"Unsupported Microsoft NNI quantization technique: {technique}"
            )
            
        quant_dtype = str(config.pop("quant_dtype", technique_dtype)).lower()
        quant_scheme = config.pop("quant_scheme", "symmetric")
        calibration_steps = int(config.pop("calibration_steps", 300))
        
        if config:
            raise TypeError(
                "Unsupported Microsoft NNI configuration options: "
                f"{', '.join(sorted(config))}"
            )
        if calibration_steps < 1:
            raise ValueError("calibration_steps must be at least 1")
        if quant_scheme not in {"symmetric", "affine"}:
            raise ValueError("quant_scheme must be 'symmetric' or 'affine'")
        if not (
            quant_dtype.startswith("int") or quant_dtype.startswith("uint")
        ):
            raise ValueError("quant_dtype must be an integer format such as int8 or uint8")
        try:
            quant_bits = int(quant_dtype.removeprefix("uint").removeprefix("int"))
        except ValueError as exc:
            raise ValueError("quant_dtype must include a bit width, such as int8") from exc
        if not 1 <= quant_bits <= 32:
            raise ValueError("quant_dtype bit width must be between 1 and 32")

        if calibration_dataset is None:
            if dummy_input is None:
                raise ValueError(
                    "NNI PTQ requires calibration_dataset or dummy_input."
                )
            calibration_dataset = [dummy_input]
            calibration_steps = 1
        elif isinstance(calibration_dataset, (torch.Tensor, dict, tuple)):
            calibration_dataset = [calibration_dataset]

        if config_list is None:
            config_list = [
                {
                    "op_types": ["Linear", "Conv1d", "Conv2d"],
                    "target_names": ["weight", "_input_0", "_output_0"],
                    "quant_dtype": quant_dtype,
                    "quant_scheme": quant_scheme,
                }
            ]
        elif not isinstance(config_list, list):
            raise TypeError("config_list must be a list of NNI quantization configs")

        quantized_model = model if inplace else deepcopy(model)
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

        def calibration_training_func(
            calibrated_model: Any,
            optimizers: Any,
            training_step: Any,
            lr_schedulers: Any = None,
            max_steps: int | None = None,
            max_epochs: int | None = None,
        ) -> None:
            was_training = calibrated_model.training
            calibrated_model.eval()
            samples_tracked = 0
            try:
                with torch.no_grad():
                    for sample in calibration_dataset:
                        sample = move_to_device(sample)
                        if isinstance(sample, dict):
                            calibrated_model(**sample)
                        elif isinstance(sample, (tuple, list)):
                            calibrated_model(*sample)
                        else:
                            calibrated_model(sample)
                        samples_tracked += 1
                        if samples_tracked >= calibration_steps:
                            break
            finally:
                calibrated_model.train(was_training)
            if samples_tracked == 0:
                raise ValueError("calibration_dataset yielded no calibration samples")

        optimizer = nni.trace(torch.optim.SGD)(quantized_model.parameters(), lr=0.0)
        evaluator = TorchEvaluator(
            training_func=calibration_training_func,
            optimizers=optimizer,
            training_step=lambda batch: None,
            dummy_input=dummy_input,
            evaluating_func=lambda evaluated_model: 0.0,
        )
        quantizer = nni_quantization.PtqQuantizer(
            quantized_model,
            config_list,
            evaluator,
        )
        quantized_model, _ = quantizer.compress(calibration_steps, None)
        return quantized_model
    