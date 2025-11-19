"""Base service class for business logic."""

import logging
from typing import Optional


class BaseService:
    """Base class for all service classes.

    Services are responsible for:
    - Pure business logic (no Qt dependencies except in return types)
    - Data transformations and calculations
    - Validation rules
    - Coordination with other services

    Services should be:
    - Framework-agnostic (can be used from CLI or GUI)
    - Fully unit testable
    - Stateless (state should be in models, not services)

    All service classes should inherit from this base class to ensure
    consistent behavior and logging support.
    """

    def __init__(self, logger_name: Optional[str] = None):
        """Initialize the base service.

        Args:
            logger_name: Optional name for the logger. If not provided,
                        uses the class name.
        """
        if logger_name is None:
            logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

    def _validate_not_none(self, value: any, param_name: str) -> None:
        """Validate that a parameter is not None.

        Args:
            value: Value to validate
            param_name: Name of the parameter (for error message)

        Raises:
            ValueError: If value is None
        """
        if value is None:
            raise ValueError(f"{param_name} cannot be None")

    def _validate_positive(self, value: float, param_name: str) -> None:
        """Validate that a numeric parameter is positive.

        Args:
            value: Value to validate
            param_name: Name of the parameter (for error message)

        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")

    def _validate_non_negative(self, value: float, param_name: str) -> None:
        """Validate that a numeric parameter is non-negative.

        Args:
            value: Value to validate
            param_name: Name of the parameter (for error message)

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError(f"{param_name} cannot be negative, got {value}")
