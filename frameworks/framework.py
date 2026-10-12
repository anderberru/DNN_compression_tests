from abc import ABC, abstractmethod
from typing import Any
from .calibrate_model import calibrate_model

class Framework(ABC):
    """Base compression framework interface.

    Subclasses must implement the `compress` method.
    """

    def __init__(self, technique: str, **kwargs: Any) -> None:
        self.technique = technique
        self.config = kwargs

    @abstractmethod
    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        """Compress a model using the requested technique.

        Args:
            model: The model object to compress.
            technique: The compression technique to use.
            **kwargs: Optional additional configuration parameters.

        Returns:
            The compressed model or compression result.
        """
        raise NotImplementedError("Subclasses must implement compress()")
