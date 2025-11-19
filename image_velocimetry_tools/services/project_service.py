"""ProjectService - Handles project save/load operations.

This service encapsulates all business logic for:
- Serializing application state to project dictionary
- Saving project dictionary to JSON
- Creating project ZIP archives
- Loading projects from ZIP archives
- Validating project data

The UI layer (ivy.py) retains responsibility for:
- Showing file dialogs
- Progress bar updates
- User confirmation dialogs
"""

import os
import json
import zipfile
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List


class ProjectService:
    """Service for project save/load operations."""

    def __init__(self):
        """Initialize the project service."""
        self.logger = logging.getLogger(__name__)

    def save_project_to_json(
        self,
        project_dict: Dict[str, Any],
        json_path: str
    ) -> bool:
        """Save project dictionary to JSON file.

        Args:
            project_dict: Dictionary containing project state
            json_path: Path where JSON file should be written

        Returns:
            True if successful, False otherwise

        Raises:
            IOError: If file cannot be written
            ValueError: If project_dict is invalid
        """
        if not isinstance(project_dict, dict):
            raise ValueError("project_dict must be a dictionary")

        if not json_path:
            raise ValueError("json_path cannot be empty")

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Write JSON with indentation for readability
            with open(json_path, "w") as fp:
                json.dump(project_dict, fp, indent=4)

            self.logger.info(f"Project saved to: {json_path}")
            return True

        except (IOError, OSError) as e:
            self.logger.error(f"Failed to save project JSON: {e}")
            raise IOError(f"Could not write project file: {e}")

    def create_project_archive(
        self,
        source_directory: str,
        output_zip_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        exclude_extensions: Optional[List[str]] = None
    ) -> bool:
        """Create a ZIP archive of the project directory.

        Args:
            source_directory: Directory to archive (swap directory)
            output_zip_path: Path for output ZIP file
            progress_callback: Optional callback(current, total) for progress updates
            exclude_extensions: List of file extensions to exclude (e.g., ['.dat'])

        Returns:
            True if successful, False otherwise

        Raises:
            IOError: If ZIP cannot be created
            FileNotFoundError: If source directory doesn't exist
        """
        if not os.path.exists(source_directory):
            raise FileNotFoundError(
                f"Source directory does not exist: {source_directory}"
            )

        if not output_zip_path:
            raise ValueError("output_zip_path cannot be empty")

        exclude_extensions = exclude_extensions or []

        try:
            # Count total items for progress tracking
            file_count = 0
            for root, _, files in os.walk(source_directory):
                for file in files:
                    # Check if file should be excluded
                    if not any(file.endswith(ext) for ext in exclude_extensions):
                        file_count += 1

            items_processed = 0

            # Create ZIP archive
            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(source_directory):
                    for file in files:
                        # Skip excluded extensions
                        if any(file.endswith(ext) for ext in exclude_extensions):
                            continue

                        file_path = os.path.join(root, file)
                        # Calculate relative path for archive
                        arcname = os.path.relpath(file_path, source_directory)

                        zipf.write(file_path, arcname=arcname)
                        items_processed += 1

                        # Report progress if callback provided
                        if progress_callback:
                            progress_callback(items_processed, file_count)

            self.logger.info(
                f"Created project archive: {output_zip_path} "
                f"({file_count} files)"
            )
            return True

        except (IOError, OSError, zipfile.BadZipFile) as e:
            self.logger.error(f"Failed to create project archive: {e}")
            raise IOError(f"Could not create project archive: {e}")

    def extract_project_archive(
        self,
        zip_path: str,
        extract_to_directory: str
    ) -> bool:
        """Extract a project ZIP archive.

        Args:
            zip_path: Path to ZIP file
            extract_to_directory: Directory where files should be extracted

        Returns:
            True if successful, False otherwise

        Raises:
            FileNotFoundError: If ZIP file doesn't exist
            IOError: If extraction fails
            zipfile.BadZipFile: If ZIP file is corrupted
        """
        if not extract_to_directory:
            raise ValueError("extract_to_directory cannot be empty")

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP file does not exist: {zip_path}")

        try:
            # Ensure extraction directory exists
            os.makedirs(extract_to_directory, exist_ok=True)

            # Extract all files
            with zipfile.ZipFile(zip_path, "r") as zipf:
                zipf.extractall(extract_to_directory)

            self.logger.info(f"Extracted project archive to: {extract_to_directory}")
            return True

        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid or corrupted ZIP file: {e}")
            raise zipfile.BadZipFile(f"Project file is corrupted: {e}")

        except (IOError, OSError) as e:
            self.logger.error(f"Failed to extract project archive: {e}")
            raise IOError(f"Could not extract project archive: {e}")

    def load_project_from_json(self, json_path: str) -> Dict[str, Any]:
        """Load project dictionary from JSON file.

        Args:
            json_path: Path to project JSON file

        Returns:
            Dictionary containing project state

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            IOError: If file cannot be read
            ValueError: If JSON is invalid
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Project JSON file does not exist: {json_path}")

        try:
            with open(json_path, "r") as fp:
                project_dict = json.load(fp)

            if not isinstance(project_dict, dict):
                raise ValueError("Project JSON must contain a dictionary")

            self.logger.info(f"Loaded project from: {json_path}")
            return project_dict

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid project JSON: {e}")
            raise ValueError(f"Project file contains invalid JSON: {e}")

        except (IOError, OSError) as e:
            self.logger.error(f"Failed to load project JSON: {e}")
            raise IOError(f"Could not read project file: {e}")

    def validate_project_dict(self, project_dict: Dict[str, Any]) -> List[str]:
        """Validate project dictionary for required fields.

        Args:
            project_dict: Dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for required top-level keys
        # Note: These are the most critical fields that must exist
        # Many fields are optional and added incrementally as features are used

        # We can add more validation as needed, but for now just check basic structure
        if not isinstance(project_dict, dict):
            errors.append("Project data must be a dictionary")

        return errors

    def load_scaffold_configuration(
        self,
        scaffold_ivy_path: str,
        temp_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load configuration from a scaffold .ivy project file.

        This method extracts all necessary parameters from a scaffold project
        for batch processing. The scaffold should contain camera calibration,
        GCPs, cross-section data, and STIV processing parameters.

        Args:
            scaffold_ivy_path: Path to scaffold .ivy project file
            temp_dir: Optional temporary directory for extraction
                     (if None, creates a temporary directory)

        Returns:
            Dictionary containing scaffold configuration:
                - project_dict: Complete project dictionary from JSON
                - swap_directory: Path to extracted swap directory
                - rectification_method: Rectification method used
                - rectification_params: Parameters for rectification
                - stiv_params: STIV processing parameters
                - cross_section_data: Cross-section geometry data
                - display_units: Display units (English/Metric)
                - temp_cleanup_required: Whether temp_dir needs cleanup

        Raises:
            FileNotFoundError: If scaffold file doesn't exist
            ValueError: If scaffold is missing required data
            IOError: If extraction fails

        Notes:
            - Caller is responsible for cleaning up temp_dir if
              temp_cleanup_required is True
            - The swap_directory contains all project files (images, data, etc.)
        """
        # Validate scaffold exists
        if not os.path.exists(scaffold_ivy_path):
            raise FileNotFoundError(f"Scaffold file not found: {scaffold_ivy_path}")

        # Create or use provided temp directory
        cleanup_required = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="ivy_scaffold_")
            cleanup_required = True
        else:
            os.makedirs(temp_dir, exist_ok=True)

        self.logger.info(f"Loading scaffold from: {scaffold_ivy_path}")

        try:
            # Extract the .ivy archive
            swap_dir = os.path.join(temp_dir, "swap")
            self.extract_project_archive(scaffold_ivy_path, swap_dir)

            # Load project JSON
            json_path = os.path.join(swap_dir, "project_data.json")
            project_dict = self.load_project_from_json(json_path)

            # Validate required fields
            validation_errors = []

            # Check for rectification data
            if "rectification_method" not in project_dict:
                validation_errors.append("Missing rectification_method")

            # Check for cross-section data
            if "cross_section_line" not in project_dict:
                validation_errors.append("Missing cross_section_line")

            # Check for STIV parameters
            stiv_required = [
                "stiv_search_line_length_m",
                "stiv_phi_origin",
                "stiv_phi_range",
                "stiv_dphi",
                "stiv_num_pixels"
            ]
            for param in stiv_required:
                if param not in project_dict:
                    validation_errors.append(f"Missing STIV parameter: {param}")

            if validation_errors:
                raise ValueError(
                    f"Scaffold validation failed:\n" + "\n".join(validation_errors)
                )

            # Extract scaffold configuration
            scaffold_config = {
                "project_dict": project_dict,
                "swap_directory": swap_dir,
                "temp_cleanup_required": cleanup_required,

                # Rectification
                "rectification_method": project_dict["rectification_method"],
                "rectification_params": self._extract_rectification_params(project_dict),

                # STIV parameters
                "stiv_params": {
                    "phi_origin": project_dict["stiv_phi_origin"],
                    "phi_range": project_dict["stiv_phi_range"],
                    "dphi": project_dict["stiv_dphi"],
                    "num_pixels": project_dict["stiv_num_pixels"],
                    "gaussian_blur_sigma": project_dict.get("stiv_gaussian_blur_sigma", 0.0),
                    "max_vel_threshold_mps": project_dict.get("stiv_max_vel_threshold_mps", 10.0),
                },

                # Cross-section
                "cross_section_data": {
                    "line": project_dict.get("cross_section_line"),
                    "bathymetry_filename": project_dict.get("bathymetry_ac3_filename"),
                    "start_bank": project_dict.get("cross_section_start_bank", "left"),
                },

                # Grid parameters
                "grid_params": {
                    "num_points": project_dict.get("number_grid_points_along_xs_line", 25),
                },

                # Units
                "display_units": project_dict.get("display_units", "English"),
            }

            self.logger.info("Scaffold configuration loaded successfully")
            self.logger.debug(
                f"Rectification method: {scaffold_config['rectification_method']}, "
                f"STIV params: {len(scaffold_config['stiv_params'])} parameters, "
                f"Grid points: {scaffold_config['grid_params']['num_points']}"
            )

            return scaffold_config

        except Exception as e:
            # Cleanup temp directory if we created it and there was an error
            if cleanup_required and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

    def _extract_rectification_params(self, project_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract rectification parameters from project dict.

        Args:
            project_dict: Project dictionary

        Returns:
            Dictionary of rectification parameters appropriate for the method
        """
        method = project_dict.get("rectification_method")

        if method == "homography":
            return {
                "homography_matrix": project_dict.get("homography_matrix"),
                "world_coords": project_dict.get("orthotable_world_coordinates"),
                "pixel_coords": project_dict.get("orthotable_pixel_coordinates"),
                "pad_x": project_dict.get("ortho_pad_x", 0),
                "pad_y": project_dict.get("ortho_pad_y", 0),
            }

        elif method == "camera matrix":
            return {
                "camera_matrix": project_dict.get("camera_matrix"),
                "water_surface_elevation": project_dict.get("water_surface_elevation_m"),
                "extent": project_dict.get("ortho_extent"),
            }

        elif method == "scale":
            return {
                "world_coords": project_dict.get("orthotable_world_coordinates"),
                "pixel_coords": project_dict.get("orthotable_pixel_coordinates"),
            }

        else:
            return {}
