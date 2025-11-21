"""Service for loading and validating scaffold project files.

This service handles loading scaffold .ivy files (ZIP archives containing
project configuration) and validating that they contain all required data
for batch processing.
"""

import os
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.services.project_service import ProjectService
from image_velocimetry_tools.batch.exceptions import InvalidScaffoldError


class ScaffoldLoader(BaseService):
    """Service for loading and validating scaffold project files.

    A scaffold project file (.ivy) contains all common configuration shared
    across batch jobs:
    - Rectification parameters (camera matrix, GCPs)
    - Cross-section geometry (AC3 file)
    - Grid generation settings
    - STIV processing parameters
    - Video processing settings (FFmpeg effects, distortion correction)

    The scaffold is loaded once and merged with job-specific parameters
    (video path, water surface elevation, etc.) for each batch job.

    Notes
    -----
    For batch processing, the following requirements apply:
    - Rectification method must be "camera_matrix" (3D calibration)
    - STIV method will be "exhaustive" (two_dimensional_stiv_exhaustive)
    - Grid must be along cross-section line with masks
    """

    # Required top-level keys in project_data.json
    REQUIRED_KEYS = [
        "rectification_parameters",
        "cross_section_geometry_path",
        "grid_parameters",
        "stiv_parameters",
        "ffmpeg_parameters",
    ]

    # Required rectification parameters
    REQUIRED_RECTIFICATION_KEYS = [
        "method",
        "ground_control_points",
        "image_control_points",
    ]

    # Required STIV parameters
    REQUIRED_STIV_KEYS = [
        "num_pixels",
        "phi_origin",
        "d_phi",
        "phi_range",
        "max_vel_threshold_mps",
    ]

    def __init__(self):
        """Initialize the ScaffoldLoader service."""
        super().__init__()
        self.project_service = ProjectService()

    def load_scaffold(
        self,
        scaffold_path: str,
        temp_extract_dir: str = None
    ) -> Dict[str, Any]:
        """Load and validate a scaffold project file.

        Parameters
        ----------
        scaffold_path : str
            Path to the scaffold .ivy file
        temp_extract_dir : str, optional
            Directory to extract scaffold to. If None, creates a temp directory.

        Returns
        -------
        dict
            Dictionary containing:
            - project_data: The loaded project_data.json configuration
            - extract_dir: Path where scaffold was extracted
            - cross_section_path: Path to AC3 cross-section file
            - calibration_image_path: Path to calibration image (if present)

        Raises
        ------
        InvalidScaffoldError
            If scaffold file is missing, corrupted, or missing required data

        Notes
        -----
        The extract_dir is NOT automatically cleaned up to allow access to
        files during batch processing. Caller is responsible for cleanup.
        """
        self.logger.info(f"Loading scaffold: {scaffold_path}")

        # Validate scaffold file exists
        scaffold_path_obj = Path(scaffold_path)
        if not scaffold_path_obj.exists():
            raise InvalidScaffoldError(
                f"Scaffold file does not exist: {scaffold_path}"
            )

        if not scaffold_path_obj.suffix == ".ivy":
            raise InvalidScaffoldError(
                f"Scaffold file must have .ivy extension, "
                f"got: {scaffold_path_obj.suffix}"
            )

        # Create extraction directory
        if temp_extract_dir is None:
            temp_extract_dir = tempfile.mkdtemp(prefix="scaffold_")
            self.logger.debug(f"Created temp directory: {temp_extract_dir}")
        else:
            os.makedirs(temp_extract_dir, exist_ok=True)

        # Extract the .ivy archive
        try:
            self.project_service.extract_project_archive(
                zip_path=scaffold_path,
                extract_to_directory=temp_extract_dir
            )
        except Exception as e:
            raise InvalidScaffoldError(
                f"Failed to extract scaffold archive: {e}"
            ) from e

        # Load project_data.json
        project_data_path = os.path.join(temp_extract_dir, "project_data.json")
        if not os.path.exists(project_data_path):
            raise InvalidScaffoldError(
                f"Scaffold is missing project_data.json file"
            )

        try:
            project_data = self.project_service.load_project_from_json(
                project_data_path
            )
        except Exception as e:
            raise InvalidScaffoldError(
                f"Failed to load project_data.json: {e}"
            ) from e

        # Validate scaffold data
        validation_errors = self.validate_scaffold_data(project_data, temp_extract_dir)
        if validation_errors:
            error_summary = "\n".join(validation_errors)
            raise InvalidScaffoldError(
                f"Scaffold validation failed:\n{error_summary}"
            )

        # Find cross-section file
        cross_section_path = self._find_cross_section_file(temp_extract_dir)
        if not cross_section_path:
            raise InvalidScaffoldError(
                "Scaffold is missing cross-section geometry file (*.mat)"
            )

        # Find calibration image (optional)
        calibration_image_path = self._find_calibration_image(temp_extract_dir)

        self.logger.info(
            f"Successfully loaded scaffold from: {scaffold_path}"
        )

        return {
            "project_data": project_data,
            "extract_dir": temp_extract_dir,
            "cross_section_path": cross_section_path,
            "calibration_image_path": calibration_image_path,
        }

    def validate_scaffold_data(
        self,
        project_data: Dict[str, Any],
        extract_dir: str
    ) -> List[str]:
        """Validate that scaffold contains all required data for batch processing.

        Parameters
        ----------
        project_data : dict
            The loaded project_data.json dictionary
        extract_dir : str
            Directory where scaffold was extracted (for file validation)

        Returns
        -------
        list of str
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for required top-level keys
        for key in self.REQUIRED_KEYS:
            if key not in project_data:
                errors.append(f"Missing required key: '{key}'")

        # Validate rectification parameters
        if "rectification_parameters" in project_data:
            rect_params = project_data["rectification_parameters"]

            # Check for required rectification keys
            for key in self.REQUIRED_RECTIFICATION_KEYS:
                if key not in rect_params:
                    errors.append(
                        f"Missing required rectification parameter: '{key}'"
                    )

            # Verify rectification method is camera_matrix
            if rect_params.get("method") != "camera matrix":
                errors.append(
                    f"Batch processing requires rectification method "
                    f"'camera matrix', got: '{rect_params.get('method')}'"
                )

            # Validate GCPs
            gcps = rect_params.get("ground_control_points", [])
            icps = rect_params.get("image_control_points", [])
            if len(gcps) < 6:
                errors.append(
                    f"Camera matrix method requires at least 6 GCPs, "
                    f"got {len(gcps)}"
                )
            if len(gcps) != len(icps):
                errors.append(
                    f"Number of GCPs ({len(gcps)}) must match number of "
                    f"ICPs ({len(icps)})"
                )

        # Validate STIV parameters
        if "stiv_parameters" in project_data:
            stiv_params = project_data["stiv_parameters"]

            for key in self.REQUIRED_STIV_KEYS:
                if key not in stiv_params:
                    errors.append(
                        f"Missing required STIV parameter: '{key}'"
                    )

        # Validate grid parameters
        if "grid_parameters" in project_data:
            grid_params = project_data["grid_parameters"]

            # Check that grid is along cross-section
            if not grid_params.get("use_cross_section_line", False):
                errors.append(
                    "Batch processing requires grid to be along cross-section line"
                )

        # Check for cross-section geometry path
        if "cross_section_geometry_path" in project_data:
            xs_path = project_data["cross_section_geometry_path"]
            if xs_path:
                full_path = os.path.join(extract_dir, xs_path)
                if not os.path.exists(full_path):
                    errors.append(
                        f"Cross-section geometry file not found: {xs_path}"
                    )

        return errors

    def _find_cross_section_file(self, extract_dir: str) -> str:
        """Find cross-section geometry file (.mat) in extracted scaffold.

        Parameters
        ----------
        extract_dir : str
            Directory where scaffold was extracted

        Returns
        -------
        str or None
            Path to cross-section file, or None if not found
        """
        # Look in 5-discharge directory (standard location)
        discharge_dir = os.path.join(extract_dir, "5-discharge")
        if os.path.exists(discharge_dir):
            for file in os.listdir(discharge_dir):
                if file.endswith(".mat") and "cross" in file.lower():
                    return os.path.join(discharge_dir, file)

        # Fallback: search entire scaffold
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(".mat") and "cross" in file.lower():
                    return os.path.join(root, file)

        return None

    def _find_calibration_image(self, extract_dir: str) -> str:
        """Find calibration image in extracted scaffold.

        Parameters
        ----------
        extract_dir : str
            Directory where scaffold was extracted

        Returns
        -------
        str or None
            Path to calibration image, or None if not found
        """
        # Look in 2-orthorectification directory
        ortho_dir = os.path.join(extract_dir, "2-orthorectification")
        if os.path.exists(ortho_dir):
            for file in os.listdir(ortho_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(ortho_dir, file)

        return None

    def cleanup_scaffold(self, extract_dir: str) -> None:
        """Clean up extracted scaffold directory.

        Parameters
        ----------
        extract_dir : str
            Directory to remove

        Notes
        -----
        This should be called when batch processing is complete to clean up
        temporary files.
        """
        if os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
                self.logger.debug(f"Cleaned up scaffold directory: {extract_dir}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to cleanup scaffold directory {extract_dir}: {e}"
                )

    def get_scaffold_info(self, scaffold_path: str) -> Dict[str, Any]:
        """Get basic information about a scaffold file without full extraction.

        Parameters
        ----------
        scaffold_path : str
            Path to the scaffold .ivy file

        Returns
        -------
        dict
            Dictionary containing:
            - is_valid: Boolean indicating if file appears valid
            - has_project_data: Boolean indicating if project_data.json exists
            - has_cross_section: Boolean indicating if cross-section file found
            - file_size_mb: File size in megabytes

        Raises
        ------
        InvalidScaffoldError
            If scaffold file cannot be accessed
        """
        import zipfile

        scaffold_path_obj = Path(scaffold_path)

        if not scaffold_path_obj.exists():
            raise InvalidScaffoldError(
                f"Scaffold file does not exist: {scaffold_path}"
            )

        info = {
            "is_valid": False,
            "has_project_data": False,
            "has_cross_section": False,
            "file_size_mb": scaffold_path_obj.stat().st_size / (1024 * 1024),
        }

        try:
            with zipfile.ZipFile(scaffold_path, 'r') as zipf:
                namelist = zipf.namelist()

                # Check for project_data.json
                info["has_project_data"] = "project_data.json" in namelist

                # Check for cross-section file
                info["has_cross_section"] = any(
                    name.endswith(".mat") and "cross" in name.lower()
                    for name in namelist
                )

                # Valid if has both
                info["is_valid"] = (
                    info["has_project_data"] and info["has_cross_section"]
                )

        except zipfile.BadZipFile:
            raise InvalidScaffoldError(
                f"Scaffold file is corrupted or not a valid ZIP: {scaffold_path}"
            )
        except Exception as e:
            raise InvalidScaffoldError(
                f"Failed to read scaffold file: {scaffold_path}. Error: {e}"
            )

        return info
