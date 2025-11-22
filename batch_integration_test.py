#!/usr/bin/env python3
"""Integration test for batch processor.

This script tests the batch processor with real data from batch_test_data/.
"""

import sys
import shutil
import platform
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from image_velocimetry_tools.services.batch_processor import BatchProcessor


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
        tz = parts[3]  # Should be "CST"
        discharge_cfs = float(parts[4])

        # Parse datetime from results file
        dt_result = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")

        # Calculate time difference
        time_diff = abs((dt_result - dt_cst).total_seconds())

        if closest_time_diff is None or time_diff < closest_time_diff:
            closest_time_diff = time_diff
            closest_discharge = discharge_cfs

    return closest_discharge


def main():
    """Run integration test."""
    print("=" * 80)
    print("BATCH PROCESSOR INTEGRATION TEST")
    print("=" * 80)

    # Setup paths
    test_data_dir = Path(__file__).parent / "batch_test_data"
    inputs_dir = test_data_dir / "inputs"
    output_dir = test_data_dir / "integration_test_output"

    # Prepare test data
    scaffold_path = inputs_dir / "scaffold_project.ivy"
    batch_csv_path = inputs_dir / "batch_boneyard.csv"

    # Create videos subdirectory and symlink video files
    videos_dir = inputs_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    video_files = [
        "03337000_bullet_20170630-115500.mp4",
        "03337000_bullet_20170630-120000.mp4",
        "03337000_bullet_20170711-103500.mp4"
    ]

    print("\n1. Setting up test data...")
    is_windows = platform.system() == "Windows"

    for video_file in video_files:
        src = inputs_dir / video_file
        dst = videos_dir / video_file
        if not dst.exists() and src.exists():
            if is_windows:
                # Copy files on Windows (symlinks require admin privileges)
                shutil.copy2(src, dst)
                print(f"   Copied: {video_file}")
            else:
                # Use symlinks on Unix (saves space)
                dst.symlink_to(src)
                print(f"   Linked: {video_file}")

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"\n2. Input files:")
    print(f"   Scaffold: {scaffold_path}")
    print(f"   Batch CSV: {batch_csv_path}")
    print(f"   Output dir: {output_dir}")

    # USGS results file with true discharge values
    results_file = inputs_dir / "03337000_results.txt"
    print(f"   USGS data: {results_file}")

    try:
        # Create batch processor
        print("\n3. Initializing BatchProcessor...")
        processor = BatchProcessor(
            scaffold_path=str(scaffold_path),
            batch_csv_path=str(batch_csv_path),
            output_dir=str(output_dir),
            stop_on_first_failure=False,
            save_ivy_projects=False  # Enable .ivy project archiving
        )

        # Run batch processing
        print("\n4. Running batch processing (with .ivy project saving)...")
        print("-" * 80)
        jobs = processor.run()
        print("-" * 80)

        # Analyze results
        print("\n5. Results Summary:")
        print("-" * 80)

        completed = [j for j in jobs if j.status.value == "completed"]
        failed = [j for j in jobs if j.status.value == "failed"]

        print(f"   Total jobs: {len(jobs)}")
        print(f"   Completed: {len(completed)}")
        print(f"   Failed: {len(failed)}")

        # Check for .ivy files
        ivy_files = list(output_dir.glob("**/*.ivy"))
        print(f"   .ivy files created: {len(ivy_files)}")
        if ivy_files:
            total_size_mb = sum(f.stat().st_size for f in ivy_files) / (1024 * 1024)
            print(f"   Total .ivy size: {total_size_mb:.1f} MB")

        if failed:
            print("\n   Failed jobs:")
            for job in failed:
                print(f"   - {job.job_id}: {job.error_message}")

        print("\n6. Discharge Comparison (vs USGS true values):")
        print("-" * 80)
        print(f"{'Video':<45} {'True (cfs)':<12} {'IVy (cfs)':<12} {'Error %':<12} {'Status'}")
        print("-" * 80)

        all_pass = True
        for job in completed:
            video_name = Path(job.video_path).stem

            # Load true discharge from USGS results file
            try:
                true_discharge = load_true_discharge(results_file, video_name)
                actual_discharge = job.discharge_value

                if true_discharge is not None:
                    error_percent = ((actual_discharge - true_discharge) / true_discharge) * 100

                    # Accept within 10% as specified by user
                    status = "✓ PASS" if abs(error_percent) <= 10.0 else "✗ FAIL"
                    if status == "✗ FAIL":
                        all_pass = False

                    print(f"{video_name:<45} {true_discharge:>10.2f}  {actual_discharge:>10.2f}  "
                          f"{error_percent:>+10.1f}%  {status}")
                else:
                    print(f"{video_name:<45} {'N/A':<12} {actual_discharge:>10.2f}  {'N/A':<12}  ?")
                    all_pass = False
            except Exception as e:
                print(f"{video_name:<45} ERROR: {str(e)}")
                all_pass = False

        # Summary
        print("\n" + "=" * 80)
        if all_pass and len(failed) == 0:
            print("✓ INTEGRATION TEST PASSED!")
            print("=" * 80)
            return 0
        else:
            print("✗ INTEGRATION TEST FAILED")
            print("=" * 80)
            return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup videos directory (symlinks on Unix, copies on Windows)
        print("\n7. Cleaning up...")
        if videos_dir.exists():
            for item in videos_dir.iterdir():
                if item.is_symlink() or item.is_file():
                    item.unlink()
            if not any(videos_dir.iterdir()):
                videos_dir.rmdir()


if __name__ == "__main__":
    sys.exit(main())
