"""Helper functions for headless batch processing

This module contains utility functions for processing videos without the GUI,
used by the batch processor to extract frames, run STIV, calculate discharge, etc.
"""

import glob
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from areacomp.gui.areasurvey import AreaSurvey
from scipy.interpolate import interp2d

from image_velocimetry_tools.common_functions import (
    hhmmss_to_seconds,
    seconds_to_hhmmss,
    calculate_uv_components,
    geographic_to_arithmetic,
)
from image_velocimetry_tools.discharge_tools import (
    compute_discharge_midsection,
    convert_surface_velocity_rantz,
)
from image_velocimetry_tools.ffmpeg_tools import create_ffmpeg_command
from image_velocimetry_tools.file_management import deserialize_numpy_array
from image_velocimetry_tools.image_processing_tools import create_grayscale_image_stack
from image_velocimetry_tools.stiv import two_dimensional_stiv_exhaustive
from image_velocimetry_tools.uncertainty import Uncertainty


def extract_frames_headless(
    video_path: str,
    output_directory: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    frame_step: int = 1,
    frame_rate: Optional[float] = None,
    max_frames: int = 200,
    ffmpeg_cmd: str = "ffmpeg"
) -> Tuple[List[str], Dict]:
    """Extract frames from video using ffmpeg (headless)

    Args:
        video_path: Path to input video file
        output_directory: Directory to save extracted frames
        start_time: Optional start time (HH:MM:SS or seconds)
        end_time: Optional end time (HH:MM:SS or seconds)
        frame_step: Extract every Nth frame (1 = all frames)
        frame_rate: Optional frame rate (will auto-detect if not provided)
        max_frames: Maximum number of frames to extract
        ffmpeg_cmd: Path to ffmpeg executable

    Returns:
        Tuple of (list of frame file paths, metadata dict)
    """
    logging.info(f"Extracting frames from: {video_path}")

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Get video metadata using ffprobe
    metadata = get_video_metadata(video_path)

    if frame_rate is None:
        frame_rate = metadata.get("avg_frame_rate", 30.0)

    # Build ffmpeg parameters
    params = {
        "input_video": video_path,
        "extract_frames": True,
        "extracted_frames_folder": output_directory,
        "extract_frame_pattern": "f%05d.jpg",
        "extract_frame_step": frame_step,
    }

    if start_time:
        # Convert to HH:MM:SS format if needed
        if ":" not in start_time:
            start_time = seconds_to_hhmmss(float(start_time))
        params["start_time"] = start_time

    if end_time:
        if ":" not in end_time:
            end_time = seconds_to_hhmmss(float(end_time))
        params["end_time"] = end_time

    # Build ffmpeg command
    ffmpeg_command = create_ffmpeg_command(params)
    logging.debug(f"FFmpeg command: {ffmpeg_command}")

    # Execute ffmpeg
    try:
        result = subprocess.run(
            ffmpeg_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logging.info("Frame extraction complete")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg extraction failed: {e.stderr.decode()}")
        raise

    # Get list of extracted frames
    frame_pattern = os.path.join(output_directory, "f*.jpg")
    frame_files = sorted(glob.glob(frame_pattern))

    # Limit to max_frames if needed
    if len(frame_files) > max_frames:
        logging.warning(f"Extracted {len(frame_files)} frames, limiting to {max_frames}")
        frame_files = frame_files[:max_frames]

    logging.info(f"Extracted {len(frame_files)} frames to {output_directory}")

    # Calculate timestep
    timestep_ms = (1000.0 / frame_rate) * frame_step

    extraction_metadata = {
        "num_frames": len(frame_files),
        "frame_rate": frame_rate,
        "frame_step": frame_step,
        "timestep_ms": timestep_ms,
        "start_time": start_time or "00:00:00",
        "end_time": end_time or "",
        "output_directory": output_directory,
    }

    return frame_files, extraction_metadata


def get_video_metadata(video_path: str, ffprobe_cmd: str = "ffprobe") -> Dict:
    """Get video metadata using ffprobe

    Args:
        video_path: Path to video file
        ffprobe_cmd: Path to ffprobe executable

    Returns:
        Dictionary with video metadata
    """
    cmd = [
        ffprobe_cmd,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        output = json.loads(result.stdout)
        stream = output['streams'][0]

        # Parse frame rate (e.g., "30000/1001" -> 29.97)
        frame_rate_str = stream.get('r_frame_rate', '30/1')
        num, denom = map(int, frame_rate_str.split('/'))
        avg_frame_rate = num / denom if denom != 0 else 30.0

        metadata = {
            "width": int(stream.get('width', 1920)),
            "height": int(stream.get('height', 1080)),
            "avg_frame_rate": avg_frame_rate,
            "duration": float(stream.get('duration', 0.0)),
            "frame_count": int(stream.get('nb_frames', 0)),
        }

        return metadata

    except Exception as e:
        logging.warning(f"Failed to get video metadata: {e}")
        # Return defaults
        return {
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": 30.0,
            "duration": 0.0,
            "frame_count": 0,
        }


def generate_cross_section_grid(
    cross_section_line: np.ndarray,
    num_points: int,
    pixel_ground_scale_distance_m: float
) -> np.ndarray:
    """Generate grid points along cross-section line

    Args:
        cross_section_line: Array of cross-section endpoints [[x1,y1], [x2,y2]]
        num_points: Number of points to generate along the line
        pixel_ground_scale_distance_m: Pixel to meter conversion factor

    Returns:
        Array of grid points (pixel coordinates) shape (num_points, 2)
    """
    # Extract endpoints
    line = cross_section_line.reshape(2, 2)
    x1, y1 = line[0]
    x2, y2 = line[1]

    # Generate points along line
    t = np.linspace(0, 1, num_points)
    x_points = x1 + t * (x2 - x1)
    y_points = y1 + t * (y2 - y1)

    grid = np.column_stack([x_points, y_points])

    logging.debug(f"Generated {num_points} grid points along cross-section")
    return grid


def orthorectify_frames_headless(
    frame_directory: str,
    projection_matrix: np.ndarray,
    water_surface_elevation_m: float,
    extent: Optional[np.ndarray] = None
) -> List[str]:
    """Orthorectify frames using camera matrix (headless)

    Args:
        frame_directory: Directory containing f*.jpg frames
        projection_matrix: Camera projection matrix (3x4 or 4x3)
        water_surface_elevation_m: Water surface elevation in meters (Z coordinate)
        extent: Optional bounding box [x_min, x_max, y_min, y_max] for orthorectification

    Returns:
        List of paths to rectified frames (t*.jpg files)
    """
    from image_velocimetry_tools.orthorectification import CameraHelper
    from skimage.io import imread
    from PIL import Image

    logging.info("Orthorectifying frames...")

    # Get list of frames to rectify
    frame_files = sorted(glob.glob(os.path.join(frame_directory, "f*.jpg")))
    if len(frame_files) == 0:
        raise ValueError(f"No f*.jpg frames found in {frame_directory}")

    logging.info(f"Orthorectifying {len(frame_files)} frames at WSE = {water_surface_elevation_m} m")

    # Read first frame to initialize camera dimensions
    first_frame = imread(frame_files[0])

    # Create camera helper with first frame and set projection matrix
    camera = CameraHelper(image=first_frame)
    camera.set_camera_matrix(projection_matrix)

    # Orthorectify each frame
    rectified_files = []
    for i, frame_path in enumerate(frame_files):
        # Read frame
        frame = imread(frame_path)

        # Get top view (orthorectified) projection
        # Use skip_size_check=True to avoid checking each frame's dimensions
        rectified = camera.get_top_view_of_image(
            frame,
            extent=extent,
            Z=water_surface_elevation_m,
            skip_size_check=True
        )

        # Save rectified frame as t*.jpg
        frame_num = i + 1
        output_path = os.path.join(frame_directory, f"t{frame_num:05d}.jpg")
        img = Image.fromarray(rectified)
        img.save(output_path)
        rectified_files.append(output_path)

    logging.info(f"Created {len(rectified_files)} orthorectified frames")
    return rectified_files


def run_stiv_headless(
    frame_files: List[str],
    grid: np.ndarray,
    stiv_params: Dict,
    pixel_gsd: float,
    timestep_ms: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Run STIV processing (headless)

    Args:
        frame_files: List of frame file paths
        grid: Grid points (pixel coords) shape (N, 2)
        stiv_params: STIV parameters dict (phi_origin, phi_range, dphi, num_pixels, etc.)
        pixel_gsd: Pixel ground scale distance (m)
        timestep_ms: Time step between frames (milliseconds)

    Returns:
        Tuple of (magnitudes_mps, directions_deg)
    """
    logging.info(f"Running STIV on {len(frame_files)} frames with {len(grid)} grid points")

    # Create grayscale image stack
    logging.debug("Creating grayscale image stack...")
    image_stack = create_grayscale_image_stack(frame_files)

    # Extract STIV parameters
    phi_origin = stiv_params.get('phi_origin', 145)
    phi_range = stiv_params.get('phi_range', 15)
    dphi = stiv_params.get('dphi', 1.0)
    num_pixels = stiv_params.get('num_pixels', 44)
    sigma = stiv_params.get('gaussian_blur_sigma', 0.0)
    max_vel_threshold = stiv_params.get('max_vel_threshold_mps', 10.0)

    # Run STIV exhaustive
    logging.debug(f"Running STIV exhaustive...")
    logging.debug(f"  Grid points: {len(grid)}")
    logging.debug(f"  Phi origin: {phi_origin}°, range: {phi_range}°, dphi: {dphi}°")
    logging.debug(f"  Search line pixels: {num_pixels}")

    # Simple progress callback for logging
    class SimpleProgressCallback:
        def emit(self, value):
            if value % 10 == 0:
                logging.debug(f"  STIV progress: {value}%")

    progress_cb = SimpleProgressCallback()

    magnitudes_mps, directions_deg, stis, thetas = two_dimensional_stiv_exhaustive(
        x_origin=grid[:, 0].astype(float),
        y_origin=grid[:, 1].astype(float),
        image_stack=image_stack,
        num_pixels=num_pixels,
        phi_origin=phi_origin,
        d_phi=dphi,
        phi_range=phi_range,
        pixel_gsd=pixel_gsd,
        d_t=timestep_ms / 1000.0,  # Convert to seconds
        sigma=sigma,
        max_vel_threshold=max_vel_threshold,
        progress_signal=progress_cb,
    )

    logging.info(f"STIV complete. Mean velocity: {np.nanmean(magnitudes_mps):.3f} m/s")

    return magnitudes_mps, directions_deg


def get_pixel_xs_headless(
    grid_pixel: np.ndarray,
    cross_section_line: np.ndarray,
    xs_survey: AreaSurvey,
    water_surface_elevation_m: float,
    cross_section_start_bank: str = "left"
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel grid points to station distances and elevations (headless)

    This is a simplified headless version of CrossSectionGeometry.get_pixel_xs()

    Args:
        grid_pixel: Grid points in pixel coordinates shape (N, 2)
        cross_section_line: Cross-section endpoints [[x1,y1], [x2,y2]]
        xs_survey: AreaSurvey object with loaded bathymetry
        water_surface_elevation_m: Water surface elevation (meters)
        cross_section_start_bank: "left" or "right"

    Returns:
        Tuple of (stations, elevations) in meters
    """
    from areacomp.gui.projectdata import ProjectData

    # Get cross-section endpoints
    point_extents = cross_section_line.reshape(2, 2)

    # Combine grid points with cross-section endpoints for projection
    new_arr = np.insert(point_extents, 1, grid_pixel, axis=0)
    df = pd.DataFrame(new_arr, columns=["x", "y"])

    # Use AreaComp's ProjectData to compute stations along the line
    proj = ProjectData()
    proj.compute_data(df, rtn=True)

    # Calculate pixel distance between cross-section endpoints
    pixel_dist = np.sqrt(
        (point_extents[1, 0] - point_extents[0, 0]) ** 2 +
        (point_extents[1, 1] - point_extents[0, 1]) ** 2
    )

    # Find wetted width from cross-section survey
    # (where the water surface intersects the cross-section)
    survey_stations = xs_survey.survey["Stations"].to_numpy()
    survey_elevations = xs_survey.survey["AdjustedStage"].to_numpy()

    # Find where WSE crosses the cross-section
    above_water = survey_elevations <= water_surface_elevation_m
    if np.any(above_water):
        crossings = survey_stations[above_water]
        wetted_width = crossings[-1] - crossings[0]
        left_edge = crossings[0]
    else:
        # No water, use full channel width
        wetted_width = np.max(survey_stations) - np.min(survey_stations)
        left_edge = np.min(survey_stations)

    # Convert pixel distances to real-world stations
    p_conversion = wetted_width / pixel_dist
    pixel_stations = proj.stations * p_conversion + left_edge

    # Handle right bank start
    if cross_section_start_bank == "right":
        pixel_stations = (
            np.nanmax(pixel_stations) -
            pixel_stations -
            (0 - np.nanmin(pixel_stations))
        )

    # Interpolate elevations for pixel stations
    elevations = np.interp(
        pixel_stations,
        survey_stations,
        survey_elevations
    )

    # Extract only the grid point results (exclude cross-section endpoints)
    # The projection included 2 endpoints + N grid points, so we extract indices 1:-1
    pixel_stations_grid = pixel_stations[1:-1]
    elevations_grid = elevations[1:-1]

    logging.debug(f"get_pixel_xs_headless: grid_pixel shape={grid_pixel.shape}, returning {len(pixel_stations_grid)} stations")

    return pixel_stations_grid, elevations_grid


def calculate_discharge_headless(
    magnitudes_mps: np.ndarray,
    directions_deg: np.ndarray,
    grid_pixel: np.ndarray,
    xs_survey: AreaSurvey,
    cross_section_line: np.ndarray,
    water_surface_elevation_m: float,
    cross_section_start_bank: str = "left",
    alpha: float = 0.85
) -> Tuple[Dict, Dict]:
    """Calculate discharge from velocity data (headless)

    Args:
        magnitudes_mps: Surface velocity magnitudes (m/s)
        directions_deg: Flow directions (degrees, geographic)
        grid_pixel: Grid points in pixel coordinates
        xs_survey: AreaSurvey cross-section survey object
        cross_section_line: Cross-section endpoints [[x1,y1], [x2,y2]]
        water_surface_elevation_m: Water surface elevation (m)
        cross_section_start_bank: "left" or "right"
        alpha: Velocity coefficient (default 0.85)

    Returns:
        Tuple of (discharge_results dict, discharge_summary dict)
    """
    logging.info("Calculating discharge...")

    # Get station distances and depths from cross-section
    stations, elevations = get_pixel_xs_headless(
        grid_pixel,
        cross_section_line,
        xs_survey,
        water_surface_elevation_m,
        cross_section_start_bank
    )
    depths = water_surface_elevation_m - elevations

    # Calculate normal velocities (perpendicular to cross-section)
    # Use magnitude_normals if available, otherwise project velocities
    D = np.radians(directions_deg)
    U = magnitudes_mps * np.cos(D)
    V = magnitudes_mps * np.sin(D)
    M = np.sqrt(U**2 + V**2)

    # Add zero velocity at edges
    surface_velocities = np.insert(M, 0, 0)
    surface_velocities = np.append(surface_velocities, 0)

    stations_with_edges = np.insert(stations, 0, stations[0])
    stations_with_edges = np.append(stations_with_edges, stations[-1])

    depths_with_edges = np.insert(depths, 0, 0)
    depths_with_edges = np.append(depths_with_edges, 0)

    # Convert surface to average velocity
    average_velocities = convert_surface_velocity_rantz(surface_velocities, alpha=alpha)

    # Debug: Check array lengths before discharge calculation
    logging.debug(f"Array lengths before discharge calculation:")
    logging.debug(f"  stations_with_edges: {len(stations_with_edges)}")
    logging.debug(f"  average_velocities: {len(average_velocities)}")
    logging.debug(f"  depths_with_edges: {len(depths_with_edges)}")

    # Compute discharge using midsection method
    total_discharge, total_area, widths, areas, unit_discharges = compute_discharge_midsection(
        stations_with_edges,
        average_velocities,
        depths_with_edges,
        return_details=True
    )

    # Build discharge results dataframe
    discharge_results = {}
    for i in range(len(stations_with_edges)):
        discharge_results[i] = {
            "ID": i,
            "Status": "Used" if not np.isnan(surface_velocities[i]) else "Not Used",
            "Station Distance": stations_with_edges[i],
            "Width": widths[i],
            "Depth": depths_with_edges[i],
            "Area": areas[i],
            "Surface Velocity": surface_velocities[i],
            "α (alpha)": alpha,
            "Unit Discharge": unit_discharges[i],
        }

    discharge_summary = {
        "total_discharge": total_discharge,
        "total_area": total_area,
        "ISO_uncertainty": None,  # Will be calculated by uncertainty module
        "IVE_uncertainty": None,
    }

    logging.info(f"Discharge calculation complete:")
    logging.info(f"  Total Q: {total_discharge:.4f} m³/s")
    logging.info(f"  Total A: {total_area:.4f} m²")
    logging.info(f"  Stations used: {sum(1 for r in discharge_results.values() if r['Status'] == 'Used')}/{len(discharge_results)}")

    return discharge_results, discharge_summary


def calculate_uncertainty_headless(
    discharge_summary: Dict,
    discharge_results: Dict,
    survey_units: Dict,
    rectification_method: str,
    num_gcp: int
) -> Dict:
    """Calculate discharge uncertainty (ISO and IVE methods)

    Args:
        discharge_summary: Discharge summary dict with total_discharge, total_area
        discharge_results: Discharge results dict (stations)
        survey_units: Survey units conversion dict
        rectification_method: "camera matrix" or "homography"
        num_gcp: Number of ground control points

    Returns:
        Dictionary with u_iso and u_ive uncertainty components
    """
    logging.debug("Calculating uncertainty...")

    try:
        # Convert discharge_results to DataFrame
        df = pd.DataFrame.from_dict(discharge_results, orient='index')

        # Initialize uncertainty calculator
        uncertainty = Uncertainty()
        uncertainty.survey_units = survey_units

        # Set discharge data
        uncertainty.total_discharge = discharge_summary["total_discharge"]
        uncertainty.total_area = discharge_summary["total_area"]
        uncertainty.discharge_data = df

        # Calculate uncertainties
        u_iso, u_iso_contribution = uncertainty.calculate_iso()
        u_ive, u_ive_contribution = uncertainty.calculate_ive()

        # Update discharge summary
        discharge_summary["ISO_uncertainty"] = u_iso["u95_q"]
        discharge_summary["IVE_uncertainty"] = u_ive["u95_q"]

        logging.info(f"  ISO uncertainty: {u_iso['u95_q']*100:.2f}%")
        logging.info(f"  IVE uncertainty: {u_ive['u95_q']*100:.2f}%")

        return {
            "u_iso": u_iso,
            "u_iso_contribution": u_iso_contribution,
            "u_ive": u_ive,
            "u_ive_contribution": u_ive_contribution,
        }

    except Exception as e:
        logging.warning(f"Failed to calculate uncertainty: {e}")
        return {
            "u_iso": {},
            "u_iso_contribution": {},
            "u_ive": {},
            "u_ive_contribution": {},
        }
