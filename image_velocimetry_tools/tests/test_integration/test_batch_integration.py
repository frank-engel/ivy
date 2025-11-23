"""Integration tests for batch processor.

This module tests the batch processor with real data from batch_test_data/.
Tests validate end-to-end batch processing including:
- Video processing
- Discharge calculations
- Comparison against USGS reference data
- .ivy project archiving
"""

import shutil
import platform
from pathlib import Path
from datetime import datetime, timedelta

import pytest

from image_velocimetry_tools.services.batch_processor import BatchProcessor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def test_data_dir():
    """Get the batch test data directory."""
    return Path(__file__).parent / "batch_test_data"


@pytest.fixture(scope="module")
def inputs_dir(test_data_dir):
    """Get the inputs directory."""
    return test_data_dir / "inputs"


@pytest.fixture(scope="module")
def output_dir(test_data_dir):
    """Create and cleanup output directory."""
    output = test_data_dir / "integration_test_output"

    # Clean before tests
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    yield output

    # Cleanup after tests (optional - comment out to inspect results)
    # if output.exists():
    #     shutil.rmtree(output)


@pytest.fixture(scope="module")
def videos_dir(inputs_dir):
    """Create videos directory with symlinked/copied video files."""
    videos_dir = inputs_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    video_files = [
        "03337000_bullet_20170630-115500.mp4",
        "03337000_bullet_20170630-120000.mp4",
        "03337000_bullet_20170711-103500.mp4"
    ]

    is_windows = platform.system() == "Windows"

    for video_file in video_files:
        src = inputs_dir / video_file
        dst = videos_dir / video_file

        if not dst.exists() and src.exists():
            if is_windows:
                # Copy files on Windows (symlinks require admin privileges)
                shutil.copy2(src, dst)
            else:
                # Use symlinks on Unix (saves space)
                dst.symlink_to(src)

    yield videos_dir

    # Cleanup: remove symlinks/copies
    if videos_dir.exists():
        for item in videos_dir.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()
        if not any(videos_dir.iterdir()):
            videos_dir.rmdir()


@pytest.fixture(scope="module")
def batch_processor(inputs_dir, output_dir, videos_dir):
    """Create and run batch processor."""
    scaffold_path = inputs_dir / "scaffold_project.ivy"
    batch_csv_path = inputs_dir / "batch_boneyard.csv"

    processor = BatchProcessor(
        scaffold_path=str(scaffold_path),
        batch_csv_path=str(batch_csv_path),
        output_dir=str(output_dir),
        stop_on_first_failure=False,
        save_ivy_projects=False  # Set to True to test .ivy archiving
    )

    # Run batch processing
    jobs = processor.run()

    return processor, jobs


@pytest.fixture(scope="module")
def usgs_results_file(inputs_dir):
    """Get USGS results file with true discharge values."""
    return inputs_dir / "03337000_results.txt"


# ============================================================================
# Helper Functions
# ============================================================================

def parse_video_timestamp(video_filename):
    """Parse timestamp from video filename.

    Args:
        video_filename: e.g., "03337000_bullet_20170630-115500"

    Returns:
        datetime object in CST timezone (video is in CDT)
    """
    # Extract timestamp from filename: YYYYMMDD-HHMMSS
    parts = video_filename.split('_')
    timestamp_str = parts[-1]  # e.g., "20170630-115500"

    # Parse datetime (this is in CDT - Central Daylight Time)
    dt_cdt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")

    # Convert CDT to CST (CDT = CST + 1 hour, so subtract 1 hour)
    dt_cst = dt_cdt - timedelta(hours=1)

    return dt_cst


def load_true_discharge(results_file, video_filename):
    """Load true discharge from USGS results file for given video.

    Args:
        results_file: Path to 03337000_results.txt
        video_filename: Video filename stem (without extension)

    Returns:
        Discharge in cfs (cubic feet per second)
    """
    # Parse video timestamp to CST
    dt_cst = parse_video_timestamp(video_filename)

    # Read results file and find closest timestamp
    with open(results_file, 'r') as f:
        lines = f.readlines()

    # Skip header
    lines = lines[1:]

    closest_discharge = None
    closest_time_diff = None

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue

        # Parse: agency_cd, site_no, datetime, tz_cd, Discharge_cfs, Approval
        datetime_str = parts[2]  # "YYYY-MM-DD HH:MM"
        discharge_cfs = float(parts[4])

        # Parse datetime from results file
        dt_result = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")

        # Calculate time difference
        time_diff = abs((dt_result - dt_cst).total_seconds())

        if closest_time_diff is None or time_diff < closest_time_diff:
            closest_time_diff = time_diff
            closest_discharge = discharge_cfs

    return closest_discharge


# ============================================================================
# Tests
# ============================================================================

def test_batch_processor_initialization(inputs_dir, output_dir, videos_dir):
    """Test that batch processor can be initialized with valid inputs."""
    scaffold_path = inputs_dir / "scaffold_project.ivy"
    batch_csv_path = inputs_dir / "batch_boneyard.csv"

    assert scaffold_path.exists(), f"Scaffold project not found: {scaffold_path}"
    assert batch_csv_path.exists(), f"Batch CSV not found: {batch_csv_path}"

    processor = BatchProcessor(
        scaffold_path=str(scaffold_path),
        batch_csv_path=str(batch_csv_path),
        output_dir=str(output_dir),
        stop_on_first_failure=False,
        save_ivy_projects=False
    )

    assert processor is not None
    assert processor.batch_jobs is not None
    assert len(processor.batch_jobs) > 0, "No batch jobs loaded from CSV"


def test_batch_processing_completes(batch_processor):
    """Test that batch processing completes without fatal errors."""
    processor, jobs = batch_processor

    assert jobs is not None, "Batch processor returned no jobs"
    assert len(jobs) > 0, "No jobs were processed"

    # Check that we have at least some completed jobs
    completed = [j for j in jobs if j.status.value == "completed"]
    assert len(completed) > 0, "No jobs completed successfully"


def test_all_jobs_complete_successfully(batch_processor):
    """Test that all jobs complete without failures."""
    processor, jobs = batch_processor

    failed = [j for j in jobs if j.status.value == "failed"]

    if failed:
        failure_details = "\n".join(
            f"  - {job.job_id}: {job.error_message}" for job in failed
        )
        pytest.fail(f"Some jobs failed:\n{failure_details}")


def test_discharge_values_exist(batch_processor):
    """Test that completed jobs have discharge values."""
    processor, jobs = batch_processor

    completed = [j for j in jobs if j.status.value == "completed"]

    for job in completed:
        assert job.discharge_value is not None, \
            f"Job {job.job_id} completed but has no discharge value"
        assert job.discharge_value > 0, \
            f"Job {job.job_id} has invalid discharge value: {job.discharge_value}"


def test_discharge_accuracy_vs_usgs(batch_processor, usgs_results_file):
    """Test that computed discharge values match USGS reference data within 10%."""
    processor, jobs = batch_processor

    assert usgs_results_file.exists(), f"USGS results file not found: {usgs_results_file}"

    completed = [j for j in jobs if j.status.value == "completed"]
    assert len(completed) > 0, "No completed jobs to validate"

    failures = []

    for job in completed:
        video_name = Path(job.video_path).stem

        try:
            true_discharge = load_true_discharge(usgs_results_file, video_name)
            actual_discharge = job.discharge_value

            if true_discharge is not None:
                error_percent = ((actual_discharge - true_discharge) / true_discharge) * 100

                # Accept within 10% as specified
                if abs(error_percent) > 10.0:
                    failures.append(
                        f"{video_name}: True={true_discharge:.2f} cfs, "
                        f"IVy={actual_discharge:.2f} cfs, Error={error_percent:+.1f}%"
                    )
        except Exception as e:
            failures.append(f"{video_name}: Error loading USGS data: {str(e)}")

    if failures:
        failure_msg = "\n  ".join(failures)
        pytest.fail(f"Discharge accuracy check failed:\n  {failure_msg}")


def test_output_directory_structure(batch_processor, output_dir):
    """Test that output directory has expected structure."""
    processor, jobs = batch_processor

    assert output_dir.exists(), "Output directory not created"

    # Check that output directory is not empty
    output_contents = list(output_dir.iterdir())
    assert len(output_contents) > 0, "Output directory is empty"

    # Check for job subdirectories
    completed = [j for j in jobs if j.status.value == "completed"]
    for job in completed:
        job_dir = output_dir / job.job_id
        assert job_dir.exists(), f"Job output directory not found: {job_dir}"


def test_discharge_output_files_created(batch_processor, output_dir):
    """Test that discharge output files are created for completed jobs."""
    processor, jobs = batch_processor

    completed = [j for j in jobs if j.status.value == "completed"]

    for job in completed:
        job_dir = output_dir / job.job_id

        # Check for discharge CSV file
        discharge_files = list(job_dir.glob("**/discharge_*.csv"))
        assert len(discharge_files) > 0, \
            f"No discharge CSV files found for job {job.job_id}"


@pytest.mark.parametrize("save_projects", [True, False])
def test_ivy_project_archiving(inputs_dir, test_data_dir, save_projects):
    """Test .ivy project archiving functionality."""
    scaffold_path = inputs_dir / "scaffold_project.ivy"
    batch_csv_path = inputs_dir / "batch_boneyard.csv"
    output_dir = test_data_dir / f"test_output_ivy_{save_projects}"

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    try:
        # Create videos directory
        videos_dir = inputs_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        processor = BatchProcessor(
            scaffold_path=str(scaffold_path),
            batch_csv_path=str(batch_csv_path),
            output_dir=str(output_dir),
            stop_on_first_failure=False,
            save_ivy_projects=save_projects
        )

        jobs = processor.run()

        # Check for .ivy files
        ivy_files = list(output_dir.glob("**/*.ivy"))

        if save_projects:
            assert len(ivy_files) > 0, ".ivy project saving enabled but no .ivy files created"

            # Verify .ivy files are valid zip archives
            for ivy_file in ivy_files:
                assert ivy_file.stat().st_size > 0, f".ivy file is empty: {ivy_file}"
        else:
            assert len(ivy_files) == 0, \
                f".ivy project saving disabled but {len(ivy_files)} .ivy files were created"

    finally:
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_job_metadata(batch_processor):
    """Test that job metadata is properly populated."""
    processor, jobs = batch_processor

    completed = [j for j in jobs if j.status.value == "completed"]

    for job in completed:
        # Check required metadata fields
        assert job.job_id is not None, "Job missing job_id"
        assert job.video_path is not None, "Job missing video_path"
        assert job.status is not None, "Job missing status"
        assert job.discharge_value is not None, "Completed job missing discharge_value"

        # Check that video path points to actual video
        video_path = Path(job.video_path)
        assert video_path.suffix == ".mp4", f"Unexpected video format: {video_path.suffix}"


def test_timestamp_parsing():
    """Test video timestamp parsing helper function."""
    # Test case: 03337000_bullet_20170630-115500
    video_name = "03337000_bullet_20170630-115500"
    dt = parse_video_timestamp(video_name)

    # Verify correct parsing and CDT to CST conversion
    assert dt.year == 2017
    assert dt.month == 6
    assert dt.day == 30
    assert dt.hour == 10  # 11:55 CDT = 10:55 CST
    assert dt.minute == 55
    assert dt.second == 0


def test_usgs_discharge_loading(usgs_results_file):
    """Test USGS discharge data loading."""
    assert usgs_results_file.exists(), "USGS results file not found"

    # Test loading discharge for a known video
    video_name = "03337000_bullet_20170630-115500"
    discharge = load_true_discharge(usgs_results_file, video_name)

    assert discharge is not None, "Failed to load discharge from USGS data"
    assert discharge > 0, f"Invalid discharge value: {discharge}"
