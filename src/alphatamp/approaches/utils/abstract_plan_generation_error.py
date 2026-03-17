"""Raised when the explorer class cannot generate an abstract plan."""


class AbstractPlanGenerationError(RuntimeError):
    """Raised when the explorer class cannot generate an abstract plan."""

    def __init__(self, message: str):
        super().__init__(f"{message}")
