"""
SettingsController - Application settings and preferences coordination

This controller manages the settings dialog, units conversions,
and application-wide preferences.
"""

import logging
import os
from image_velocimetry_tools.gui.controllers.base_controller import BaseController
from image_velocimetry_tools.gui.models.settings_model import SettingsModel
from image_velocimetry_tools.gui.settings_dialog import Settings_Dialog
from image_velocimetry_tools.common_functions import units_conversion, resource_path


class SettingsController(BaseController):
    """Controller for managing application settings and preferences."""

    def __init__(self, main_window, settings_model: SettingsModel, sticky_settings):
        """
        Initialize the SettingsController.

        Args:
            main_window: Reference to the main IVyTools window
            settings_model: SettingsModel instance
            sticky_settings: Settings instance for persistence
        """
        super().__init__(main_window, "SettingsController")
        self.settings_model = settings_model
        self.sticky_settings = sticky_settings

    # ==================== Settings Dialog ====================

    def open_settings_dialog(self):
        """
        Open the settings dialog.

        Allows the user to change the display units for the survey.
        If the units have changed, it updates the display units,
        changes the plot settings accordingly, and converts any relevant data to the new units.
        """
        mw = self.main_window

        # If the user had a previous display units, use it
        try:
            ss = self.sticky_settings.get("last_display_units")
            self.settings_model.display_units = ss
        except KeyError:
            self.sticky_settings.new("last_display_units", self.settings_model.display_units)

        # Open settings dialog
        dialog = Settings_Dialog(
            units=self.settings_model.display_units,
            icon_path=resource_path(
                resource_path(mw.__icon_path__ + os.sep + "IVy_Logo.ico"),
            ),
            parent=mw,
        )
        dialog.exec_()

        # Check if units changed
        if dialog.units is not None:
            if dialog.units != self.settings_model.display_units:
                old_units = self.settings_model.display_units
                self.settings_model.display_units = dialog.units

                # Update file menu
                mw.actionUnits.setText(f"Units: {self.settings_model.units_label}")

                # Update the user's settings
                self.sticky_settings.set("last_display_units", self.settings_model.display_units)

                # Apply units conversions
                self.change_units(self.settings_model.display_units)

                self.logger.info(
                    f"Display units changed from {old_units} to {self.settings_model.display_units}"
                )

    # ==================== Units Conversion ====================

    def change_units(self, units: str = "English"):
        """
        Apply units conversions and labels globally to all loaded data and elements.

        Args:
            units: The units to be applied ("English" or "Metric")

        This function updates labels and converts values for:
        - Water Surface Elevation
        - Pixel GSD
        - STIV velocity thresholds
        - Cross-section measurements
        - Orthorectification table
        - Cross-section geometry (AC3)
        - Discharge calculations
        """
        mw = self.main_window
        survey_units = self.settings_model.survey_units

        # Update GUI Labels
        mw.labelWaterSurfaceElevation.setText(
            f"Water Surface Elevation {survey_units['label_L']}:"
        )
        mw.labelPixelGSD.setText(f"Pixel GSD {survey_units['label_L']}:")
        mw.labelStivMaxVelThreshold.setText(
            f"Max Vel. Threshold {survey_units['label_V']}:"
        )
        mw.labelStivOptMaxVelThreshold.setText(
            f"Max Vel. Threshold {survey_units['label_V']}:"
        )
        mw.labelCrossSectionMeasurementStage.setText(
            f"Measurement Stage {survey_units['label_L']}:"
        )
        mw.stationStationLabel.setText(
            f"Starting Station {survey_units['label_L']}:"
        )
        mw.gageHeightLabel.setText(f"Stage {survey_units['label_L']}:")

        # Orthorectification Points Table
        if mw.is_ortho_table_loaded:
            mw.orthotable_populate_table(mw.orthotable_dataframe)
            mw.orthotable_change_units()

        # Water Surface Elevation spinbox
        old = mw.doubleSpinBoxRectificationWaterSurfaceElevation.value()
        mw.doubleSpinBoxRectificationWaterSurfaceElevation.setValue(
            old * units_conversion(units)['L']
        )

        # Pixel GSD
        if mw.pixel_ground_scale_distance_m is not None:
            old = float(mw.lineeditPixelGSD.text())
            mw.lineeditPixelGSD.setText(
                f"{old * units_conversion(units)['L']:.3f}"
            )

        # Cross-section Geometry Tab (AC3)
        if mw.bathymetry_ac3_filename is not None:
            # Reloading will change units
            mw.xs_survey.load_areacomp(mw.bathymetry_ac3_filename)

            # Then delete the old subsurvey
            survey = mw.xs_survey.xs_survey.surveys[-1].file_id
            mw.xs_survey.xs_survey.remove_survey(survey)
            mw.xs_survey.update_backend()

            # Update the AC3 gui elements and backend
            old = mw.char_stage_sb.value()
            mw.char_stage_sb.setValue(old * units_conversion(units)["L"])
            old = float(mw.start_station_lineEdit.text())
            mw.start_station_lineEdit.setText(
                f"{old * units_conversion(units)['L']:.3f}"
            )

            # Update XS Survey with new units
            mw.xs_survey.change = {
                "start_sta": float(mw.start_station_lineEdit.text()),
                "default_wse": mw.char_stage_sb.value(),
                "default_wse_unit": units,
                "stations_unit": units,
                "elevations_unit": units,
            }
            mw.xs_survey.update_ui()

        # Discharge Tab - update table if it exists
        if hasattr(mw, 'discharge_summary') and mw.discharge_summary:
            # The discharge tab will update when it's accessed
            pass

        self.logger.debug(f"Units conversion applied: {units}")

    # ==================== Initialization ====================

    def initialize_units_from_settings(self):
        """Initialize display units from sticky settings."""
        try:
            ss = self.sticky_settings.get("last_display_units")
            self.settings_model.display_units = ss
            self.logger.debug(f"Loaded display units from settings: {ss}")
        except KeyError:
            self.settings_model.display_units = "English"
            self.sticky_settings.new("last_display_units", "English")
            self.logger.debug("Initialized display units to English (default)")

        return self.settings_model.display_units
