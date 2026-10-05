from typing import Any

from .framework import Framework


class ExampleFramework(Framework):
    """Example framework implementation that inherits Framework."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Compress a model using the requested technique."""
        if technique != self.technique:
            raise ValueError(
                f"Technique mismatch: expected {self.technique!r}, got {technique!r}"
            )

        if technique not in {"prune", "pruning"}:
            raise ValueError(
                "TensorFlow Model Optimization only supports the 'prune' technique"
            )

        try:
            import tensorflow_model_optimization as tfmot
        except ImportError as exc:
            raise RuntimeError(
                "TF-MOT pruning requires tensorflow-model-optimization. "
                "Install it with `pip install tensorflow-model-optimization`."
            ) from exc

        config = {**self.config, **kwargs}
        config.pop("dummy_input", None)
        initial_sparsity = float(config.pop("initial_sparsity", 0.0))
        final_sparsity = float(config.pop("final_sparsity", 0.5))
        begin_step = int(config.pop("begin_step", 0))
        end_step = int(config.pop("end_step", 1000))
        frequency = int(config.pop("frequency", 100))

        if not 0.0 <= initial_sparsity <= 1.0:
            raise ValueError("initial_sparsity must be between 0.0 and 1.0")
        if not 0.0 <= final_sparsity <= 1.0:
            raise ValueError("final_sparsity must be between 0.0 and 1.0")
        if initial_sparsity > final_sparsity:
            raise ValueError("initial_sparsity cannot exceed final_sparsity")
        if begin_step < 0 or end_step <= begin_step:
            raise ValueError("end_step must be greater than begin_step >= 0")
        if frequency < 1:
            raise ValueError("frequency must be at least 1")

        pruning_schedule = config.pop("pruning_schedule", None)
        if pruning_schedule is None:
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=initial_sparsity,
                final_sparsity=final_sparsity,
                begin_step=begin_step,
                end_step=end_step,
                frequency=frequency,
            )

        return tfmot.sparsity.keras.prune_low_magnitude(
            model,
            pruning_schedule=pruning_schedule,
            **config,
        )
