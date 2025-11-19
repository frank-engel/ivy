#!/usr/bin/env python
"""Debug script for batch processing with detailed logging.

This script runs a single video through batch processing with debug
logging enabled to diagnose rectification issues.
"""

import logging
import sys
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_debug.log', mode='w')
    ]
)

# Add parent directory to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from image_velocimetry_tools.api import process_video

def main():
    """Run single video with debug logging."""

    scaffold_path = repo_root / "examples" / "scaffold_project.ivy"
    video_path = repo_root / "examples" / "videos" / "03337000_bullet_20170630-120000.mp4"
    output_dir = repo_root / "batch_output" / "debug_test"

    print("=" * 70)
    print("Batch Processing Debug Test")
    print("=" * 70)
    print(f"\nScaffold: {scaffold_path}")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"\nLogging to: batch_debug.log")
    print("=" * 70)
    print()

    # Progress callback
    def progress_callback(percent, message):
        print(f"[{percent:3d}%] {message}")

    try:
        result = process_video(
            scaffold_path=str(scaffold_path),
            video_path=str(video_path),
            water_surface_elevation=318.50,
            output_directory=str(output_dir),
            measurement_date="2017-06-30",
            alpha=0.85,
            max_frames=5,  # Only process 5 frames for faster debugging
            progress_callback=progress_callback,
            cleanup_temp_files=False,
        )

        print("\n" + "=" * 70)
        if result.success:
            print("✓ Processing completed successfully!")
            print(f"\nDischarge: {result.total_discharge:.4f} m³/s")
            print(f"Area: {result.total_area:.4f} m²")
            print(f"Velocity: {result.mean_velocity:.4f} m/s")
        else:
            print(f"✗ Processing failed at stage: {result.error_stage}")
            print(f"Error: {result.error_message}")
            print(f"\nCheck batch_debug.log for detailed information")
        print("=" * 70)

        return 0 if result.success else 1

    except Exception as e:
        print(f"\n✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nCheck batch_debug.log for detailed information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
