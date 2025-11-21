#!/usr/bin/env python3
"""Integration test for batch processor.

This script tests the batch processor with real data from batch_test_data/.
"""

import sys
import shutil
import platform
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from image_velocimetry_tools.services.batch_processor import BatchProcessor


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

    # Expected discharge values from batch_discharge_summary.csv
    expected_discharges = {
        "03337000_bullet_20170630-115500": 2.6096,
        "03337000_bullet_20170630-120000": 3.6419,
        "03337000_bullet_20170711-103500": 3.6369,
    }

    try:
        # Create batch processor
        print("\n3. Initializing BatchProcessor...")
        processor = BatchProcessor(
            scaffold_path=str(scaffold_path),
            batch_csv_path=str(batch_csv_path),
            output_dir=str(output_dir),
            stop_on_first_failure=False
        )

        # Run batch processing
        print("\n4. Running batch processing...")
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

        if failed:
            print("\n   Failed jobs:")
            for job in failed:
                print(f"   - {job.job_id}: {job.error_message}")

        print("\n6. Discharge Comparison:")
        print("-" * 80)
        print(f"{'Video':<45} {'Expected':<12} {'Actual':<12} {'Diff':<12} {'Status'}")
        print("-" * 80)

        all_pass = True
        for job in completed:
            video_name = Path(job.video_path).stem
            expected = expected_discharges.get(video_name, None)
            actual = job.discharge_value

            if expected is not None:
                diff = actual - expected
                percent_diff = (diff / expected) * 100 if expected != 0 else 0

                # Consider within 5% as acceptable (due to numerical differences)
                status = "✓ PASS" if abs(percent_diff) < 5.0 else "✗ FAIL"
                if status == "✗ FAIL":
                    all_pass = False

                print(f"{video_name:<45} {expected:>10.4f}  {actual:>10.4f}  "
                      f"{diff:>+10.4f}  {status}")
            else:
                print(f"{video_name:<45} {'N/A':<12} {actual:>10.4f}  {'N/A':<12}  ?")

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
