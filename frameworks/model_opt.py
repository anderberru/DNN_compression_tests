from typing import Any

from .framework import Framework


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

        if technique not in {"prune", "pruning"}:
            raise ValueError(
                "NVIDIA Model Optimizer only supports the 'pruning' technique"
            )


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
        return compressed_model
