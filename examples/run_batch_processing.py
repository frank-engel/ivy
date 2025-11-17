#!/usr/bin/env python
"""Example script demonstrating how to use the IVyTools batch processor

This script shows how to process multiple videos from the same fixed camera
using a scaffold .ivy project as a template.

Usage:
    python run_batch_processing.py

Requirements:
    1. A scaffold .ivy project with:
       - Camera matrix / homography calibration
       - Ground control points (GCPs)
       - Cross-section geometry (AreaComp .mat file)
       - STIV processing parameters
    2. A CSV file listing videos with their water surface elevations
    3. Video files from the same camera location
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import image_velocimetry_tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_velocimetry_tools.batch_processor import (
    BatchProcessor,
    create_batch_config_template,
)


def main():
    """Main batch processing function"""

    # ==================================================================
    # CONFIGURATION - Update these paths for your setup
    # ==================================================================

    # Get the directory where this script is located
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = script_dir  # This script is in the examples directory

    # Path to scaffold .ivy project (template with camera calibration & cross-section)
    scaffold_ivy_path = os.path.join(examples_dir, "scaffold_project.ivy")

    # Path to CSV file with batch configuration (video list + WSE)
    batch_config_csv = os.path.join(examples_dir, "batch_boneyard.csv")

    # Output directory for results
    output_directory = os.path.join(examples_dir, "output")
    os.makedirs(output_directory, exist_ok=True)

    # Configure logging to output folder
    log_file = os.path.join(output_directory, "batch_processing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log each run
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )

    # ==================================================================
    # OPTIONAL: Create a batch config template
    # ==================================================================
    # Uncomment to create a template CSV file:
    # create_batch_config_template("batch_config_template.csv")
    # print("Created batch_config_template.csv - edit this file with your video paths and WSE values")
    # return

    # ==================================================================
    # BATCH PROCESSING
    # ==================================================================

    print("="*60)
    print("IVyTools Batch Video Processor")
    print("="*60)
    print(f"Scaffold: {scaffold_ivy_path}")
    print(f"Config:   {batch_config_csv}")
    print(f"Output:   {output_directory}")
    print("="*60)

    # Progress callback (optional)
    def progress_callback(current, total, message):
        """Print progress updates"""
        print(f"[{current}/{total}] {message}")

    # Initialize batch processor
    try:
        processor = BatchProcessor(
            scaffold_ivy_path=scaffold_ivy_path,
            batch_config_csv=batch_config_csv
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize batch processor: {e}")
        return 1

    print(f"\nLoaded {len(processor.batch_configs)} videos to process")

    # Process all videos
    try:
        results = processor.process_batch(
            output_directory=output_directory,
            progress_callback=progress_callback
        )
    except Exception as e:
        print(f"ERROR: Batch processing failed: {e}")
        return 1

    # Export summary CSV
    summary_csv_path = f"{output_directory}/batch_discharge_summary.csv"
    processor.export_results_csv(results, summary_csv_path)

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)

    # Print summary
    successful = [r for r in results if r.processing_status == "success"]
    failed = [r for r in results if r.processing_status == "failed"]

    print(f"\nTotal videos:  {len(results)}")
    print(f"Successful:    {len(successful)}")
    print(f"Failed:        {len(failed)}")

    if successful:
        print("\n Successful discharges:")
        for r in successful:
            q_str = f"{r.total_discharge:.2f} m3/s" if r.total_discharge is not None else "N/A"
            unc_str = f"{r.iso_uncertainty*100:.1f}%" if r.iso_uncertainty is not None else "N/A"
            print(f"  {r.video_filename}: Q={q_str} (ISO unc: {unc_str})")

    if failed:
        print("\n✗ Failed videos:")
        for r in failed:
            print(f"  {r.video_filename}: {r.error_message}")

    print(f"\nResults saved to:")
    print(f"  Individual .ivy files: {output_directory}/")
    print(f"  Summary CSV:           {summary_csv_path}")
    print(f"  Log file:              batch_processing.log")

    # Cleanup
    processor.cleanup()

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
