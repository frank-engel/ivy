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
from typing import Dict, Any, Callable, Optional, List


class ProjectService:
    """Service for project save/load operations."""

    def __init__(self):
        """Initialize the project service."""
        self.logger = logging.getLogger(__name__)

    def save_project_to_json(
        self, project_dict: Dict[str, Any], json_path: str
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
        exclude_extensions: Optional[List[str]] = None,
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
                    if not any(
                        file.endswith(ext) for ext in exclude_extensions
                    ):
                        file_count += 1

            items_processed = 0

            # Create ZIP archive
            with zipfile.ZipFile(
                output_zip_path, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                for root, _, files in os.walk(source_directory):
                    for file in files:
                        # Skip excluded extensions
                        if any(
                            file.endswith(ext) for ext in exclude_extensions
                        ):
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
        self, zip_path: str, extract_to_directory: str
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

            self.logger.info(
                f"Extracted project archive to: {extract_to_directory}"
            )
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
            raise FileNotFoundError(
                f"Project JSON file does not exist: {json_path}"
            )

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
