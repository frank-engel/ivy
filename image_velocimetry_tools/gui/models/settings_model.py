"""
SettingsModel - Settings and preferences state management

This model holds application-wide settings including display units,
preferences, and configuration that persists across sessions.
"""

from PyQt5.QtCore import pyqtSignal
from image_velocimetry_tools.gui.models.base_model import BaseModel
from image_velocimetry_tools.common_functions import units_conversion


class SettingsModel(BaseModel):
    """Model for application settings and preferences."""

    # Qt Signals
    display_units_changed = pyqtSignal(str)  # units: "English" or "Metric"

    def __init__(self):
        super().__init__()
        # Units settings
        self._display_units: str = "English"
        self._units_label: str = "English"
        self._survey_units: dict = units_conversion(units_id="English")

    # ==================== Properties ====================

    @property
    def display_units(self) -> str:
        """Display units for the application ("English" or "Metric")."""
        return self._display_units

    @display_units.setter
    def display_units(self, value: str):
        if value not in ["English", "Metric"]:
            raise ValueError(f"Invalid display units: {value}. Must be 'English' or 'Metric'")
        if self._display_units != value:
            self._display_units = value
            self._units_label = value
            self._survey_units = units_conversion(units_id=value)
            self.display_units_changed.emit(value)

    @property
    def units_label(self) -> str:
        """Units label (same as display_units, maintained for backwards compatibility)."""
        return self._units_label

    @units_label.setter
    def units_label(self, value: str):
        if value not in ["English", "Metric"]:
            raise ValueError(f"Invalid units label: {value}. Must be 'English' or 'Metric'")
        if self._units_label != value:
            self._units_label = value
            self._display_units = value
            self._survey_units = units_conversion(units_id=value)
            self.display_units_changed.emit(value)

    @property
    def survey_units(self) -> dict:
        """Survey units conversion dictionary."""
        return self._survey_units

    # ==================== Methods ====================

    def reset(self):
        """Reset settings to default values."""
        self._display_units = "English"
        self._units_label = "English"
        self._survey_units = units_conversion(units_id="English")
        self.display_units_changed.emit("English")
