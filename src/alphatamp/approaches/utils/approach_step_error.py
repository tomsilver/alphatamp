"""Raised when any error occurs while running approach.step()."""


class ApproachStepError(BaseException):
    """Raised when any error occurs while running approach.step()."""

    def __init__(self, message: str, original_exception: BaseException):
        super().__init__(f"{message}: {original_exception}")
        self.original_exception = original_exception
