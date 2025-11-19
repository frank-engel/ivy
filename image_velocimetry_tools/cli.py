"""Command-line interface for IVyTools batch video processing.

This module provides a command-line tool for batch processing river videos
using the IVyTools workflow. The CLI wraps the Python API and provides
progress reporting, validation, and output formatting.

Usage:
    # Process a single video
    ivytools-batch process scaffold.ivy river.mp4 --wse 318.5 --output results/

    # Process batch from CSV
    ivytools-batch batch scaffold.ivy batch_config.csv --output batch_results/

    # Validate configuration
    ivytools-batch validate scaffold.ivy batch_config.csv

    # Display scaffold info
    ivytools-batch info scaffold.ivy
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional

from image_velocimetry_tools.api import (
    process_video,
    process_batch_csv,
    load_scaffold,
    ProcessingResult,
    BatchResult,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="ivytools-batch",
        description="IVyTools batch video processing for river discharge measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  ivytools-batch process scaffold.ivy video.mp4 --wse 318.5 --output results/

  # Process batch from CSV
  ivytools-batch batch scaffold.ivy batch.csv --output batch_results/

  # Validate configuration
  ivytools-batch validate scaffold.ivy batch.csv

  # Display scaffold info
  ivytools-batch info scaffold.ivy

For more information, see: https://github.com/frank-engel-usgs/IVyTools
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="IVyTools Batch Processing 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ========================================
    # Command: process
    # ========================================
    process_parser = subparsers.add_parser(
        "process",
        help="Process a single video",
        description="Process a single video through the complete workflow",
    )
    process_parser.add_argument(
        "scaffold",
        help="Path to scaffold .ivy template project",
    )
    process_parser.add_argument(
        "video",
        help="Path to input video file",
    )
    process_parser.add_argument(
        "--wse",
        type=float,
        required=True,
        help="Water surface elevation in meters",
    )
    process_parser.add_argument(
        "--output",
        "-o",
        default="output/",
        help="Output directory (default: output/)",
    )
    process_parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Alpha coefficient for velocity adjustment (default: 0.85)",
    )
    process_parser.add_argument(
        "--start",
        type=float,
        help="Start time in seconds (default: video start)",
    )
    process_parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds (default: video end)",
    )
    process_parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1)",
    )
    process_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to extract (default: no limit)",
    )
    process_parser.add_argument(
        "--date",
        help="Measurement date (YYYY-MM-DD format)",
    )
    process_parser.add_argument(
        "--comments",
        help="Optional comments about this measurement",
    )
    process_parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (frames, etc.)",
    )
    process_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    # ========================================
    # Command: batch
    # ========================================
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process batch from CSV file",
        description="Process multiple videos from CSV configuration file",
    )
    batch_parser.add_argument(
        "scaffold",
        help="Path to scaffold .ivy template project",
    )
    batch_parser.add_argument(
        "csv",
        help="Path to batch configuration CSV file",
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        default="batch_output/",
        help="Output directory (default: batch_output/)",
    )
    batch_parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (frames, etc.)",
    )
    batch_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    # ========================================
    # Command: validate
    # ========================================
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration files",
        description="Validate scaffold and batch CSV configuration",
    )
    validate_parser.add_argument(
        "scaffold",
        help="Path to scaffold .ivy template project",
    )
    validate_parser.add_argument(
        "csv",
        nargs="?",
        help="Optional path to batch CSV file",
    )

    # ========================================
    # Command: info
    # ========================================
    info_parser = subparsers.add_parser(
        "info",
        help="Display scaffold information",
        description="Display scaffold configuration details",
    )
    info_parser.add_argument(
        "scaffold",
        help="Path to scaffold .ivy template project",
    )

    return parser


def command_process(args) -> int:
    """Execute 'process' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Progress callback
    last_percent = -1

    def progress_callback(percent, message):
        nonlocal last_percent
        if not args.quiet and percent != last_percent:
            print(f"[{percent:3d}%] {message}")
            last_percent = percent

    # Process video
    print(f"Processing: {Path(args.video).name}")
    print(f"Output: {args.output}\n")

    try:
        result = process_video(
            scaffold_path=args.scaffold,
            video_path=args.video,
            water_surface_elevation=args.wse,
            output_directory=args.output,
            measurement_date=args.date or "",
            alpha=args.alpha,
            start_time=args.start,
            end_time=args.end,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            comments=args.comments or "",
            progress_callback=progress_callback,
            cleanup_temp_files=not args.keep_temp,
        )

        # Display results
        print("\n" + "=" * 60)
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
            return 0
        else:
            print(f"✗ Processing failed at stage: {result.error_stage}")
            print(f"  Error: {result.error_message}")
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


def command_batch(args) -> int:
    """Execute 'batch' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Progress callback
    last_percent = -1

    def progress_callback(percent, message):
        nonlocal last_percent
        if not args.quiet and percent != last_percent:
            print(f"[{percent:3d}%] {message}")
            last_percent = percent

    # Process batch
    print(f"Batch CSV: {args.csv}")
    print(f"Output: {args.output}\n")

    try:
        batch_result = process_batch_csv(
            scaffold_path=args.scaffold,
            batch_csv_path=args.csv,
            output_directory=args.output,
            progress_callback=progress_callback,
            cleanup_temp_files=not args.keep_temp,
        )

        # Display results
        print("\n" + "=" * 60)
        print(f"Batch processing complete!\n")
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
            print(f"  Mean:  {discharge_summary['mean']:.4f} m³/s")
            print(f"  Min:   {discharge_summary['min']:.4f} m³/s")
            print(f"  Max:   {discharge_summary['max']:.4f} m³/s")
            print(f"  Std:   {discharge_summary['std']:.4f} m³/s")

        # Individual results
        print(f"\nIndividual results:")
        for result in batch_result.video_results:
            video_name = Path(result.video_path).stem
            if result.success:
                print(f"  ✓ {video_name}: Q={result.total_discharge:.4f} m³/s")
            else:
                print(f"  ✗ {video_name}: {result.error_stage}")

        print(f"\nBatch summary: {batch_result.batch_csv_path}")

        return 0 if batch_result.failed == 0 else 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


def command_validate(args) -> int:
    """Execute 'validate' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = valid, 1 = invalid)
    """
    print("Validating configuration...\n")

    valid = True

    # Validate scaffold
    print("Scaffold:")
    try:
        scaffold = load_scaffold(args.scaffold)
        print(f"  ✓ Scaffold loaded: {args.scaffold}")
        print(f"    Rectification: {scaffold.rectification_method}")
        print(f"    STIV params:   {len(scaffold.stiv_params)} parameters")
        print(f"    Grid points:   {scaffold.grid_params.get('num_points', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Scaffold invalid: {e}")
        valid = False

    # Validate batch CSV if provided
    if args.csv:
        print(f"\nBatch CSV:")
        try:
            if not os.path.exists(args.csv):
                raise FileNotFoundError(f"CSV not found: {args.csv}")

            import csv
            with open(args.csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                raise ValueError("CSV is empty")

            # Check required columns
            required = ["video_path", "water_surface_elevation"]
            for col in required:
                if col not in rows[0]:
                    raise ValueError(f"Missing required column: {col}")

            print(f"  ✓ CSV loaded: {args.csv}")
            print(f"    Videos: {len(rows)}")

            # Check video files exist
            missing = []
            for row in rows:
                if not os.path.exists(row["video_path"]):
                    missing.append(row["video_path"])

            if missing:
                print(f"  ⚠ Warning: {len(missing)} video files not found")
                for path in missing[:3]:  # Show first 3
                    print(f"    - {path}")
                if len(missing) > 3:
                    print(f"    ... and {len(missing) - 3} more")
            else:
                print(f"    All video files exist")

        except Exception as e:
            print(f"  ✗ CSV invalid: {e}")
            valid = False

    # Summary
    print("\n" + "=" * 60)
    if valid:
        print("✓ Configuration is valid")
        return 0
    else:
        print("✗ Configuration has errors")
        return 1


def command_info(args) -> int:
    """Execute 'info' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print(f"Scaffold information: {args.scaffold}\n")

    try:
        scaffold = load_scaffold(args.scaffold)

        print("Configuration:")
        print(f"  Rectification method: {scaffold.rectification_method}")
        print(f"  Display units:        {scaffold.display_units}")

        print(f"\nSTIV parameters:")
        print(f"  phi_origin:  {scaffold.stiv_params['phi_origin']:.1f}°")
        print(f"  phi_range:   {scaffold.stiv_params['phi_range']:.1f}°")
        print(f"  dphi:        {scaffold.stiv_params['dphi']:.2f}°")
        print(f"  num_pixels:  {scaffold.stiv_params['num_pixels']}")

        print(f"\nGrid parameters:")
        for key, value in scaffold.grid_params.items():
            print(f"  {key}: {value}")

        print(f"\nCross-section data:")
        xs_data = scaffold.cross_section_data
        print(f"  Has bathymetry: {xs_data.get('bathymetry_filename') is not None}")
        print(f"  Has line:       {xs_data.get('line') is not None}")

        print(f"\nRectification parameters:")
        for key, value in list(scaffold.rectification_params.items())[:5]:
            print(f"  {key}: {type(value).__name__}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == "process":
        return command_process(args)
    elif args.command == "batch":
        return command_batch(args)
    elif args.command == "validate":
        return command_validate(args)
    elif args.command == "info":
        return command_info(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
