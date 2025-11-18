"""Orthorectification state model."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from PyQt5.QtCore import pyqtSignal

from image_velocimetry_tools.gui.models.base_model import BaseModel


class OrthoModel(BaseModel):
    """Model for orthorectification state management.

    This model holds all state related to orthorectification including:
    - Ground control points (GCP) table data
    - Rectification parameters (homography/camera matrix)
    - Quality metrics (RMSE, reprojection errors)
    - Image transformation settings (flip, zoom)
    - Water surface elevation

    Signals:
        gcp_table_loaded: Emitted when GCP table is loaded (file_path: str)
        gcp_table_changed: Emitted when GCP table data changes
        gcp_image_loaded: Emitted when GCP image is loaded (file_path: str)
        rectification_calculated: Emitted when rectification parameters are calculated
        rectification_applied: Emitted when rectification is applied to images
    """

    # Qt Signals
    gcp_table_loaded = pyqtSignal(str)  # file_path
    gcp_table_changed = pyqtSignal()
    gcp_image_loaded = pyqtSignal(str)  # file_path
    rectification_calculated = pyqtSignal(str)  # method: scale/homography/camera
    rectification_applied = pyqtSignal()

    def __init__(self):
        """Initialize the orthorectification model."""
        super().__init__()

        # GCP Table Data
        self._orthotable_dataframe: pd.DataFrame = pd.DataFrame()
        self._orthotable_is_changed: bool = False
        self._is_ortho_table_loaded: bool = False
        self._orthotable_cell_colored: bool = False
        self._orthotable_has_headers: bool = False

        # GCP File Info
        self._orthotable_file_name: str = ""
        self._orthotable_fname: str = ""  # Filename without path
        self._orthotable_file_survey_units: str = "English"  # or "Metric"
        self._last_orthotable_file_name: Optional[str] = None
        self._last_ortho_gcp_image_path: Optional[str] = None

        # Rectification Parameters
        self._rectification_params: Dict[str, Any] = {
            "homography_matrix": np.eye(3),
            "camera_matrix": np.hstack((np.eye(3), np.ones((3, 1)))),
            "pixel_coords": None,
            "world_coords": None,
            "extent": None,
            "pad_x": 200,
            "pad_y": 200,
        }

        # Rectification Method and Flags
        self._rectification_method: Optional[str] = None  # "scale", "homography", "camera matrix"
        self._is_homography_matrix: bool = False
        self._is_camera_matrix: bool = False

        # Image Transformation Settings
        self._is_ortho_flip_x: bool = False
        self._is_ortho_flip_y: bool = False
        self._ortho_rectified_wse_m: float = 0.0  # Water surface elevation

        # Quality Metrics
        self._pixel_ground_scale_distance_m: Optional[float] = None
        self._scene_averaged_pixel_gsd_m: Optional[float] = None
        self._rectification_rmse_m: Optional[float] = None
        self._reprojection_error_pixels: Optional[float] = None
        self._reprojection_error_gcp_pixel_xy: Optional[np.ndarray] = None
        self._reprojection_error_gcp_pixel_total: Optional[float] = None

        # Camera Information
        self._camera_position: Optional[np.ndarray] = None
        self._rectified_transformed_gcp_points: Optional[np.ndarray] = None

        # UI State (for convenience)
        self._ortho_original_image_digitized_points: List = []
        self._ortho_original_image_zoom_factor: float = 1.0
        self._ortho_rectified_image_zoom_factor: float = 1.0
        self._ortho_original_image_current_pixel: Optional[List[int]] = None

    # ==================== GCP Table Properties ====================

    @property
    def orthotable_dataframe(self) -> pd.DataFrame:
        """Get GCP table dataframe."""
        return self._orthotable_dataframe

    @orthotable_dataframe.setter
    def orthotable_dataframe(self, value: pd.DataFrame):
        """Set GCP table dataframe."""
        if not value.equals(self._orthotable_dataframe):
            self._orthotable_dataframe = value
            self._orthotable_is_changed = True
            self.gcp_table_changed.emit()

    @property
    def orthotable_is_changed(self) -> bool:
        """Check if GCP table has unsaved changes."""
        return self._orthotable_is_changed

    @orthotable_is_changed.setter
    def orthotable_is_changed(self, value: bool):
        """Set GCP table changed flag."""
        self._orthotable_is_changed = value

    @property
    def is_ortho_table_loaded(self) -> bool:
        """Check if GCP table is loaded."""
        return self._is_ortho_table_loaded

    @is_ortho_table_loaded.setter
    def is_ortho_table_loaded(self, value: bool):
        """Set GCP table loaded flag."""
        self._is_ortho_table_loaded = value

    @property
    def orthotable_file_name(self) -> str:
        """Get GCP table file name."""
        return self._orthotable_file_name

    @orthotable_file_name.setter
    def orthotable_file_name(self, value: str):
        """Set GCP table file name."""
        self._orthotable_file_name = value

    @property
    def orthotable_file_survey_units(self) -> str:
        """Get GCP table survey units ('English' or 'Metric')."""
        return self._orthotable_file_survey_units

    @orthotable_file_survey_units.setter
    def orthotable_file_survey_units(self, value: str):
        """Set GCP table survey units."""
        self._orthotable_file_survey_units = value

    # ==================== Rectification Parameters Properties ====================

    @property
    def rectification_params(self) -> Dict[str, Any]:
        """Get rectification parameters dictionary."""
        return self._rectification_params

    @rectification_params.setter
    def rectification_params(self, value: Dict[str, Any]):
        """Set rectification parameters dictionary."""
        self._rectification_params = value

    @property
    def rectification_method(self) -> Optional[str]:
        """Get rectification method ('scale', 'homography', or 'camera matrix')."""
        return self._rectification_method

    @rectification_method.setter
    def rectification_method(self, value: str):
        """Set rectification method."""
        self._rectification_method = value

    @property
    def is_ortho_flip_x(self) -> bool:
        """Check if horizontal flip is enabled."""
        return self._is_ortho_flip_x

    @is_ortho_flip_x.setter
    def is_ortho_flip_x(self, value: bool):
        """Set horizontal flip."""
        self._is_ortho_flip_x = value

    @property
    def is_ortho_flip_y(self) -> bool:
        """Check if vertical flip is enabled."""
        return self._is_ortho_flip_y

    @is_ortho_flip_y.setter
    def is_ortho_flip_y(self, value: bool):
        """Set vertical flip."""
        self._is_ortho_flip_y = value

    @property
    def ortho_rectified_wse_m(self) -> float:
        """Get water surface elevation in meters."""
        return self._ortho_rectified_wse_m

    @ortho_rectified_wse_m.setter
    def ortho_rectified_wse_m(self, value: float):
        """Set water surface elevation in meters."""
        self._ortho_rectified_wse_m = value

    # ==================== Quality Metrics Properties ====================

    @property
    def pixel_ground_scale_distance_m(self) -> Optional[float]:
        """Get pixel ground scale distance in meters."""
        return self._pixel_ground_scale_distance_m

    @pixel_ground_scale_distance_m.setter
    def pixel_ground_scale_distance_m(self, value: float):
        """Set pixel ground scale distance."""
        self._pixel_ground_scale_distance_m = value

    @property
    def rectification_rmse_m(self) -> Optional[float]:
        """Get rectification RMSE in meters."""
        return self._rectification_rmse_m

    @rectification_rmse_m.setter
    def rectification_rmse_m(self, value: float):
        """Set rectification RMSE."""
        self._rectification_rmse_m = value

    @property
    def camera_position(self) -> Optional[np.ndarray]:
        """Get camera position in world coordinates."""
        return self._camera_position

    @camera_position.setter
    def camera_position(self, value: np.ndarray):
        """Set camera position."""
        self._camera_position = value

    # ==================== UI State Properties ====================

    @property
    def ortho_original_image_zoom_factor(self) -> float:
        """Get original image zoom factor."""
        return self._ortho_original_image_zoom_factor

    @ortho_original_image_zoom_factor.setter
    def ortho_original_image_zoom_factor(self, value: float):
        """Set original image zoom factor."""
        self._ortho_original_image_zoom_factor = value

    @property
    def ortho_rectified_image_zoom_factor(self) -> float:
        """Get rectified image zoom factor."""
        return self._ortho_rectified_image_zoom_factor

    @ortho_rectified_image_zoom_factor.setter
    def ortho_rectified_image_zoom_factor(self, value: float):
        """Set rectified image zoom factor."""
        self._ortho_rectified_image_zoom_factor = value

    # ==================== Model Methods ====================

    def reset(self):
        """Reset all orthorectification state to initial values."""
        self._orthotable_dataframe = pd.DataFrame()
        self._orthotable_is_changed = False
        self._is_ortho_table_loaded = False
        self._orthotable_cell_colored = False
        self._orthotable_has_headers = False

        self._orthotable_file_name = ""
        self._orthotable_fname = ""
        self._orthotable_file_survey_units = "English"

        self._rectification_params = {
            "homography_matrix": np.eye(3),
            "camera_matrix": np.hstack((np.eye(3), np.ones((3, 1)))),
            "pixel_coords": None,
            "world_coords": None,
            "extent": None,
            "pad_x": 200,
            "pad_y": 200,
        }

        self._rectification_method = None
        self._is_homography_matrix = False
        self._is_camera_matrix = False

        self._is_ortho_flip_x = False
        self._is_ortho_flip_y = False
        self._ortho_rectified_wse_m = 0.0

        self._pixel_ground_scale_distance_m = None
        self._scene_averaged_pixel_gsd_m = None
        self._rectification_rmse_m = None
        self._reprojection_error_pixels = None
        self._reprojection_error_gcp_pixel_xy = None
        self._reprojection_error_gcp_pixel_total = None

        self._camera_position = None
        self._rectified_transformed_gcp_points = None

        self._ortho_original_image_digitized_points = []
        self._ortho_original_image_zoom_factor = 1.0
        self._ortho_rectified_image_zoom_factor = 1.0
        self._ortho_original_image_current_pixel = None

    def get_pixel_and_world_coords(self) -> tuple:
        """Extract pixel and world coordinates from GCP table.

        Returns:
            Tuple of (pixel_coords, world_coords) as numpy arrays
        """
        if self._orthotable_dataframe.empty:
            return None, None

        # Assuming GCP table has columns: X_pixel, Y_pixel, X_world, Y_world, Z_world
        pixel_coords = self._orthotable_dataframe[["X_pixel", "Y_pixel"]].to_numpy()
        world_coords = self._orthotable_dataframe[["X_world", "Y_world", "Z_world"]].to_numpy()

        return pixel_coords, world_coords

    def update_rectification_params(self, params: Dict[str, Any]):
        """Update rectification parameters from service calculations.

        Args:
            params: Dictionary of parameters from orthorectification service
        """
        self._rectification_params.update(params)

        # Update individual quality metrics if provided
        if "pixel_gsd" in params:
            self._pixel_ground_scale_distance_m = params["pixel_gsd"]

        if "camera_position" in params:
            self._camera_position = params["camera_position"]
