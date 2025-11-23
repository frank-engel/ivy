"""Base controller class for UI coordination."""

import logging
from typing import Optional
from PyQt5.QtCore import QObject


class BaseController(QObject):
    """Base class for all controller classes.

    Controllers are responsible for:
    - UI event routing for their feature area
    - Coordination between UI widgets and services
    - UI state updates based on service responses
    - NO business logic - only orchestration

    Controllers should:
    - Accept main window, model, and service as dependencies
    - Connect UI signals to controller methods
    - Delegate business logic to services
    - Update models based on service results
    - Update UI based on model changes

    All controller classes should inherit from this base class to ensure
    consistent behavior and Qt signal support.
    """

    def __init__(
        self,
        main_window,
        model: Optional[QObject] = None,
        service: Optional[object] = None,
        logger_name: Optional[str] = None,
    ):
        """Initialize the base controller.

        Args:
            main_window: Reference to main window for widget access
            model: Optional model for state management
            service: Optional service for business logic
            logger_name: Optional name for the logger. If not provided,
                        uses the class name.
        """
        super().__init__()
        self.main_window = main_window
        self.model = model
        self.service = service

        if logger_name is None:
            logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

    def _connect_signals(self) -> None:
        """Connect UI signals to controller methods.

        Override this method in subclasses to set up signal connections.
        This method should be called during controller initialization.
        """
        pass

    def _setup_ui(self) -> None:
        """Set up initial UI state.

        Override this method in subclasses to configure initial UI state.
        This method should be called during controller initialization.
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed.

        Override this method in subclasses to perform cleanup operations
        such as disconnecting signals, stopping threads, etc.
        """
        pass
