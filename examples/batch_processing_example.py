#!/usr/bin/env python3
"""Example: Batch processing multiple videos with IVyTools.

This script demonstrates how to use the IVyTools batch processing API
to analyze multiple videos automatically. It uses the test data from
the batch_test_data folder.

The script shows three usage patterns:
1. Basic usage with minimal code
2. Usage with progress reporting
3. Advanced usage with result analysis
"""

import sys
from pathlib import Path

# Add project root to path (only needed if running script directly)
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_velocimetry_tools.api import run_batch_processing


def example_1_basic():
    """Example 1: Basic usage with minimal code."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Define paths (relative to this script)
    data_dir = Path(__file__).parent.parent / "image_velocimetry_tools/tests/test_integration/batch_test_data"
    scaffold = data_dir / "inputs" / "scaffold_project.ivy"
    batch_csv = data_dir / "inputs" / "batch_boneyard.csv"
    output = data_dir / "example_output_basic"

    # Run batch processing (simple one-liner)
    results = run_batch_processing(
        scaffold_project=str(scaffold),
        batch_csv=str(batch_csv),
        output_folder=str(output)
    )

    # Print summary
    results.print_summary()

    # Access individual results
    print("\nDetailed Results:")
    for job in results.jobs:
        if job.successful:
            print(f"  ✓ {job.video_name}:")
            print(f"      Discharge: {job.discharge:.2f} {job.discharge_units}")
            print(f"      Area: {job.area:.2f} {job.area_units}")
            print(f"      Time: {job.processing_time:.1f}s")
        else:
            print(f"  ✗ {job.video_name}: {job.error_message}")

    print()


def example_2_with_progress():
    """Example 2: Usage with progress reporting."""
    print("=" * 80)
    print("EXAMPLE 2: With Progress Reporting")
    print("=" * 80)

    # Define paths
    data_dir = Path(__file__).parent.parent / "image_velocimetry_tools/tests/test_integration/batch_test_data"
    scaffold = data_dir / "inputs" / "scaffold_project.ivy"
    batch_csv = data_dir / "inputs" / "batch_boneyard.csv"
    output = data_dir / "example_output_progress"

    # Define progress callback
    def show_progress(job_num, total_jobs, video_name, status):
        """Display progress as jobs are processed."""
        percent = (job_num / total_jobs) * 100
        bar_length = 30
        filled = int(bar_length * job_num / total_jobs)
        bar = "█" * filled + "-" * (bar_length - filled)

        print(f"\r[{bar}] {percent:>5.1f}% | Job {job_num}/{total_jobs}: {video_name} ({status})", end="")

        if job_num == total_jobs:
            print()  # New line when done

    # Run batch processing with progress callback
    print("Processing videos...")
    results = run_batch_processing(
        scaffold_project=str(scaffold),
        batch_csv=str(batch_csv),
        output_folder=str(output),
        progress_callback=show_progress
    )

    # Print summary
    print("\nProcessing complete!")
    print(f"Successful: {results.successful_jobs}/{results.total_jobs}")
    print(f"Total time: {results.total_time:.1f} seconds")
    print(f"Results saved to: {results.output_folder}")
    print()


def example_3_advanced_analysis():
    """Example 3: Advanced usage with result analysis."""
    print("=" * 80)
    print("EXAMPLE 3: Advanced Result Analysis")
    print("=" * 80)

    # Define paths
    data_dir = Path(__file__).parent.parent / "image_velocimetry_tools/tests/test_integration/batch_test_data"
    scaffold = data_dir / "inputs" / "scaffold_project.ivy"
    batch_csv = data_dir / "inputs" / "batch_boneyard.csv"
    output = data_dir / "example_output_advanced"

    # Run batch processing
    # Note: save_projects=True will create .ivy files (large, slow)
    results = run_batch_processing(
        scaffold_project=str(scaffold),
        batch_csv=str(batch_csv),
        output_folder=str(output),
        stop_on_error=False,  # Continue even if some jobs fail
        save_projects=False   # Set to True to save .ivy projects
    )

    # Analyze successful results
    successful = results.get_successful_jobs()
    if successful:
        print(f"\nAnalysis of {len(successful)} successful jobs:")
        print("-" * 80)

        # Calculate statistics
        discharges = [job.discharge for job in successful]
        areas = [job.area for job in successful]
        times = [job.processing_time for job in successful]

        avg_discharge = sum(discharges) / len(discharges)
        avg_area = sum(areas) / len(areas)
        avg_time = sum(times) / len(times)

        print(f"Average discharge: {avg_discharge:.2f} {successful[0].discharge_units}")
        print(f"Average area: {avg_area:.2f} {successful[0].area_units}")
        print(f"Average processing time: {avg_time:.1f} seconds")

        print(f"\nDischarge range: {min(discharges):.2f} - {max(discharges):.2f} {successful[0].discharge_units}")
        print(f"Area range: {min(areas):.2f} - {max(areas):.2f} {successful[0].area_units}")

        # Show detailed statistics for each job
        print(f"\nDetailed Statistics:")
        print("-" * 80)
        header = f"{'Video':<40} {'Discharge':<15} {'Area':<15} {'Q/A':<10}"
        print(header)
        print("-" * 80)

        for job in successful:
            q_over_a = job.discharge / job.area if job.area > 0 else 0
            print(f"{job.video_name:<40} "
                  f"{job.discharge:>10.2f} {job.discharge_units:<4} "
                  f"{job.area:>10.2f} {job.area_units:<4} "
                  f"{q_over_a:>8.4f}")

        # Access additional details if available
        if successful[0].details:
            print("\nAdditional statistics available in job.details:")
            example_details = successful[0].details
            for key in example_details.keys():
                print(f"  - {key}")

    # Report failures
    failed = results.get_failed_jobs()
    if failed:
        print(f"\n{len(failed)} jobs failed:")
        for job in failed:
            print(f"  ✗ {job.video_name}: {job.error_message}")

    # Find specific job by video name
    print("\nLooking up specific video...")
    specific_job = results.get_job_by_video("03337000_bullet_20170630-115500.mp4")
    if specific_job:
        print(f"Found: {specific_job}")
        if specific_job.successful:
            print(f"  Water elevation: {specific_job.water_elevation}")
            print(f"  Alpha: {specific_job.alpha}")
            print(f"  Processing time: {specific_job.processing_time:.1f}s")

    print("\n" + "=" * 80)
    print(f"All results saved to: {results.summary_csv}")
    print("=" * 80)
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "IVyTools Batch Processing Examples" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\nThis script demonstrates three ways to use the batch processing API:")
    print("  1. Basic usage - Simple and straightforward")
    print("  2. With progress - Monitor processing in real-time")
    print("  3. Advanced - Detailed result analysis")
    print("\n")

    try:
        # Run examples
        example_1_basic()
        example_2_with_progress()
        example_3_advanced_analysis()

        print("✓ All examples completed successfully!")
        print("\nNext steps:")
        print("  - Review the output folders to see generated results")
        print("  - Check batch_summary.csv for detailed statistics")
        print("  - Modify this script to use your own data")
        print()

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have the batch_test_data folder with:")
        print("  - inputs/scaffold_project.ivy")
        print("  - inputs/batch_boneyard.csv")
        print("  - inputs/videos/*.mp4")
        print()
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
