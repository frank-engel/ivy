#!/usr/bin/env python
"""Working example script for batch processing with IVyTools.

This script demonstrates complete batch processing using the real test data
from the examples/ directory. It processes the Boneyard Creek videos with
the scaffold project.

Before running:
1. Ensure IVyTools is installed: pip install -e .
2. Ensure ffmpeg and ffprobe are available (or set FFMPEG-IVyTools and FFPROBE-IVyTools env vars)
3. Run from repository root: python batch_processing_example.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for development
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from image_velocimetry_tools.api import (
    process_video,
    process_batch_csv,
    load_scaffold,
)


def example_1_inspect_scaffold():
    """Example 1: Load and inspect scaffold configuration."""
    print("=" * 70)
    print("Example 1: Inspecting Scaffold Configuration")
    print("=" * 70)

    scaffold_path = repo_root / "examples" / "scaffold_project.ivy"

    if not scaffold_path.exists():
        print(f"⚠ Scaffold not found: {scaffold_path}")
        print("Please ensure examples/ directory is present with scaffold_project.ivy")
        return False

    print(f"\nLoading scaffold: {scaffold_path.name}")

    try:
        scaffold = load_scaffold(str(scaffold_path))

        print("\n✓ Scaffold loaded successfully!")
        print(f"\nConfiguration:")
        print(f"  Rectification:   {scaffold.rectification_method}")
        print(f"  Display units:   {scaffold.display_units}")

        print(f"\nSTIV Parameters:")
        print(f"  phi_origin:  {scaffold.stiv_params['phi_origin']:.1f}°")
        print(f"  phi_range:   {scaffold.stiv_params['phi_range']:.1f}°")
        print(f"  dphi:        {scaffold.stiv_params['dphi']:.2f}°")
        print(f"  num_pixels:  {scaffold.stiv_params['num_pixels']}")

        print(f"\nGrid Parameters:")
        grid_params = scaffold.grid_params
        print(f"  num_points:  {grid_params['num_points']}")
        print(f"  pixel_gsd:   {grid_params['pixel_gsd']:.4f} m/px")

        if grid_params['grid_points'] is not None:
            print(f"  grid_points: {len(grid_params['grid_points'])} points generated")
            print(f"               from ({grid_params['grid_points'][0, 0]:.1f}, {grid_params['grid_points'][0, 1]:.1f}) "
                  f"to ({grid_params['grid_points'][-1, 0]:.1f}, {grid_params['grid_points'][-1, 1]:.1f})")
        else:
            print(f"  grid_points: None (no cross-section line)")

        print(f"\nCross-section:")
        xs_data = scaffold.cross_section_data
        print(f"  Has line:       {xs_data['line'] is not None}")
        print(f"  Has bathymetry: {xs_data.get('bathymetry_filename') is not None}")

        return True

    except Exception as e:
        print(f"\n✗ Error loading scaffold: {e}")
        return False


def example_2_process_single_video():
    """Example 2: Process a single video."""
    print("\n\n" + "=" * 70)
    print("Example 2: Processing Single Video")
    print("=" * 70)

    scaffold_path = repo_root / "examples" / "scaffold_project.ivy"
    video_path = repo_root / "examples" / "videos" / "03337000_bullet_20170630-120000.mp4"
    output_dir = repo_root / "batch_output" / "single_video"

    if not scaffold_path.exists():
        print(f"⚠ Scaffold not found: {scaffold_path}")
        return False

    if not video_path.exists():
        print(f"⚠ Video not found: {video_path}")
        print("Please ensure examples/videos/ directory contains test videos")
        return False

    print(f"\nProcessing video: {video_path.name}")
    print(f"Output directory: {output_dir}\n")

    # Progress callback
    last_percent = -1

    def progress_callback(percent, message):
        nonlocal last_percent
        if percent != last_percent:
            print(f"  [{percent:3d}%] {message}")
            last_percent = percent

    try:
        result = process_video(
            scaffold_path=str(scaffold_path),
            video_path=str(video_path),
            water_surface_elevation=318.50,  # meters (from batch CSV)
            output_directory=str(output_dir),
            measurement_date="2017-06-30",
            alpha=0.85,
            start_time=None,  # Process entire video
            end_time=None,
            frame_step=1,
            max_frames=50,  # Limit to 50 frames for faster testing
            progress_callback=progress_callback,
            cleanup_temp_files=False,  # Keep files for inspection
        )

        print("\n" + "=" * 70)
        if result.success:
            print("✓ Processing completed successfully!\n")
            print(f"Results:")
            print(f"  Discharge:       {result.total_discharge:.4f} m³/s")
            print(f"  Area:            {result.total_area:.4f} m²")
            print(f"  Mean velocity:   {result.mean_velocity:.4f} m/s")
            print(f"  Frames:          {result.num_frames_extracted}")
            print(f"  Processing time: {result.processing_time_seconds:.1f}s")

            print(f"\nOutput files:")
            print(f"  CSV:     {result.output_csv_path}")
            print(f"  Project: {result.output_project_path}")

            # Compare to expected (from BATCH_VALIDATION.md)
            expected_q = 3.1995  # m³/s for 120000 video
            tolerance = 0.5  # 0.5 m³/s tolerance (example uses different video)
            if abs(result.total_discharge - expected_q) < tolerance:
                print(f"\n✓ Discharge within expected range (±{tolerance} m³/s)")
            else:
                print(f"\n⚠ Discharge differs from expected {expected_q:.4f} m³/s")
                print(f"  (This is OK - different videos have different discharges)")

            return True
        else:
            print(f"✗ Processing failed at stage: {result.error_stage}")
            print(f"  Error: {result.error_message}")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_3_process_batch_csv():
    """Example 3: Process batch from CSV."""
    print("\n\n" + "=" * 70)
    print("Example 3: Processing Batch from CSV")
    print("=" * 70)

    scaffold_path = repo_root / "examples" / "scaffold_project.ivy"
    batch_csv_path = repo_root / "examples" / "batch_boneyard.csv"
    output_dir = repo_root / "batch_output" / "batch_csv"

    if not scaffold_path.exists():
        print(f"⚠ Scaffold not found: {scaffold_path}")
        return False

    if not batch_csv_path.exists():
        print(f"⚠ Batch CSV not found: {batch_csv_path}")
        return False

    print(f"\nBatch CSV: {batch_csv_path.name}")
    print(f"Output directory: {output_dir}\n")

    # Progress callback
    last_percent = -1

    def progress_callback(percent, message):
        nonlocal last_percent
        if percent != last_percent and percent % 5 == 0:  # Print every 5%
            print(f"  [{percent:3d}%] {message}")
            last_percent = percent

    try:
        # Note: This will process all videos in the CSV
        # You may want to limit max_frames for faster testing
        batch_result = process_batch_csv(
            scaffold_path=str(scaffold_path),
            batch_csv_path=str(batch_csv_path),
            output_directory=str(output_dir),
            progress_callback=progress_callback,
            cleanup_temp_files=False,  # Keep files for inspection
        )

        print("\n" + "=" * 70)
        print("Batch processing complete!\n")

        print(f"Summary:")
        print(f"  Total videos:    {batch_result.total_videos}")
        print(f"  Successful:      {batch_result.successful}")
        print(f"  Failed:          {batch_result.failed}")
        success_rate = (batch_result.successful / batch_result.total_videos * 100) \
            if batch_result.total_videos > 0 else 0
        print(f"  Success rate:    {success_rate:.1f}%")
        print(f"  Processing time: {batch_result.processing_time_seconds:.1f}s")

        # Discharge statistics
        discharge_summary = batch_result.get_discharge_summary()
        if discharge_summary['count'] > 0:
            print(f"\nDischarge statistics:")
            print(f"  Count: {discharge_summary['count']}")
            print(f"  Mean:  {discharge_summary['mean']:.4f} m³/s")
            print(f"  Min:   {discharge_summary['min']:.4f} m³/s")
            print(f"  Max:   {discharge_summary['max']:.4f} m³/s")
            print(f"  Std:   {discharge_summary['std']:.4f} m³/s")

        # Individual results
        print(f"\nIndividual results:")
        for result in batch_result.video_results:
            video_name = Path(result.video_path).stem
            if result.success:
                print(f"  ✓ {video_name}: Q={result.total_discharge:.4f} m³/s, "
                      f"A={result.total_area:.4f} m², V={result.mean_velocity:.4f} m/s")
            else:
                print(f"  ✗ {video_name}: {result.error_stage} - {result.error_message}")

        print(f"\nBatch summary CSV: {batch_result.batch_csv_path}")

        return batch_result.failed == 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run examples."""
    print("\n" + "=" * 70)
    print("IVyTools Batch Processing Examples")
    print("=" * 70)
    print("\nThese examples demonstrate batch processing using real test data")
    print("from the examples/ directory (Boneyard Creek videos).")
    print("\nNote: Examples will create output in batch_output/ directory")

    results = {}

    # Example 1: Inspect scaffold
    results['scaffold'] = example_1_inspect_scaffold()

    if not results['scaffold']:
        print("\n⚠ Skipping remaining examples due to scaffold loading failure")
        return 1

    # Example 2: Process single video
    print("\n\nWould you like to run Example 2 (process single video)?")
    print("This will process one video with max 50 frames (~30-60 seconds)")
    response = input("Run Example 2? [y/N]: ").strip().lower()

    if response == 'y':
        results['single'] = example_2_process_single_video()
    else:
        print("Skipping Example 2")
        results['single'] = None

    # Example 3: Process batch
    print("\n\nWould you like to run Example 3 (process batch from CSV)?")
    print("This will process all videos in batch CSV (may take several minutes)")
    response = input("Run Example 3? [y/N]: ").strip().lower()

    if response == 'y':
        results['batch'] = example_3_process_batch_csv()
    else:
        print("Skipping Example 3")
        results['batch'] = None

    # Summary
    print("\n\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)

    print("\nResults:")
    for name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        print(f"  {name:12s}: {status}")

    print("\nOutput directory: batch_output/")
    print("\nFor more information:")
    print("  - API documentation: docs/BATCH_PROCESSING.md")
    print("  - Module README: image_velocimetry_tools/batch/README.md")
    print("  - CLI help: ivytools-batch --help")

    # Return exit code
    if results.get('single') is False or results.get('batch') is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
