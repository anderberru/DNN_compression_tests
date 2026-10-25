
from typing import Any


def calibrate_model(model: Any, dummy_input: Any, calibration_steps: int = 1) -> None:
    """Run calibration forward passes using the configured dummy input."""
    import torch
    import inspect

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

    num_params = len(inspect.signature(model.forward).parameters)

    was_training = model.training
    model.eval()
    inputs = move_to_device(dummy_input, device)

    if isinstance(inputs, (tuple, list)):
        inputs = tuple(inputs[:num_params])
        
    with torch.no_grad():
        for _ in range(calibration_steps):
            if isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
    model.train(was_training)

