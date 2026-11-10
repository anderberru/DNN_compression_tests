from typing import Any

from .framework import Framework
from .calibrate_model import calibrate_model


class Uncompressed(Framework):
    """Example framework implementation that inherits Framework."""

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        #################################
        # Import necessary modules here
        #################################

        return model
