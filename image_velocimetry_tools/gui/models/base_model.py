"""Base model class for state management."""

from typing import Any, Dict
from PyQt5.QtCore import QObject, pyqtSignal


class BaseModel(QObject):
    """Base class for all model classes.

    Models are responsible for:
    - Holding application state
    - Emitting signals when state changes
    - Validating state transitions
    - Serializing/deserializing state

    All model classes should inherit from this base class to ensure
    consistent behavior and Qt signal support.

    Signals:
        state_changed: Emitted when any model state changes
    """

    state_changed = pyqtSignal(str, object)  # (property_name, new_value)

    def __init__(self):
        """Initialize the base model."""
        super().__init__()

    def _emit_state_change(self, property_name: str, new_value: Any) -> None:
        """Emit state change signal.

        Args:
            property_name: Name of the property that changed
            new_value: New value of the property
        """
        self.state_changed.emit(property_name, new_value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state to dictionary.

        Override this method in subclasses to provide custom serialization.

        Returns:
            Dictionary representation of model state
        """
        return {}

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserialize model state from dictionary.

        Override this method in subclasses to provide custom deserialization.

        Args:
            data: Dictionary containing model state
        """
        pass

    def reset(self) -> None:
        """Reset model to initial state.

        Override this method in subclasses to provide custom reset logic.
        """
        pass
