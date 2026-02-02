"""Raised when the controller runs out of actions during plan execution."""


class ControllerActionsExhaustedError(IndexError):
    """Raised when the controller runs out of actions during plan execution."""

    def __init__(self, message: str):
        super().__init__(f"{message}")
