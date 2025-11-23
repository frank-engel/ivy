"""IVy module containing video processing functions using FFMPEG

FFMPEG License: GNU Lesser General Public License (LGPL) v2.1
   (https://ffmpeg.org/legal.html)
FFMPEG Builds are here: https://www.gyan.dev/ffmpeg/builds/
"""

import logging
import os
import re
import subprocess
import json
from pathlib import Path
import shutil

from image_velocimetry_tools.common_functions import (
    hhmmss_to_seconds,
    resource_path,
    quotify_a_string,
)

# Priority: ENV var > System PATH > local ./bin fallback
ffmpeg_cmd = os.environ.get("FFMPEG-IVyTools") or shutil.which("ffmpeg")
ffprobe_cmd = os.environ.get("FFPROBE-IVyTools") or shutil.which("ffprobe")
IVY_ENV = os.environ.get("IVY_ENV")

if IVY_ENV == "development":
    fallback_path = Path(resource_path("bin"))
    ffmpeg_fallback = fallback_path / "ffmpeg.exe"
    ffprobe_fallback = fallback_path / "ffprobe.exe"

    if ffmpeg_fallback.exists() and ffprobe_fallback.exists():
        ffmpeg_cmd = str(ffmpeg_fallback)[:-4]
        ffprobe_cmd = str(ffprobe_fallback)[:-4]
        logging.warning(
            "[ffmpeg] Using local ./bin/ fallback; system binaries not found."
        )
else:
    if not ffmpeg_cmd or not ffprobe_cmd:
        fallback_path = Path(resource_path("bin"))
        ffmpeg_fallback = fallback_path / "ffmpeg.exe"
        ffprobe_fallback = fallback_path / "ffprobe.exe"

        if ffmpeg_fallback.exists() and ffprobe_fallback.exists():
            ffmpeg_cmd = str(ffmpeg_fallback)[:-4]
            ffprobe_cmd = str(ffprobe_fallback)[:-4]
            logging.warning(
                "[ffmpeg] Using local ./bin/ fallback; system binaries not found."
            )
        else:
            raise FileNotFoundError(
                "FFmpeg and FFprobe not found in environment variables, system PATH, or local ./bin/"
            )


def ffprobe_frame_count(file_path):
    """Count video frames using ffprobe.

    Returns
    -------
    int or None
        Frame count if successful, None otherwise.
    """

    if not os.path.exists(file_path):
        logging.warning(
            f"[ffprobe_frame_count] File does not exist: {file_path}"
        )
        return None

    cmd = [
        ffprobe_cmd,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-print_format",
        "json",
        file_path,
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        output = json.loads(result.stdout)
        frame_count = int(output["streams"][0]["nb_read_frames"])
        return frame_count
    except Exception as e:
        logging.warning(
            f"[ffprobe_frame_count] ffprobe failed for {file_path} with error: {e}"
        )
        return None


def ffmpeg_compute_motion_trajectories_from_frames_command(
    images_folder,
    file_pattern="f%05d.jpg",
    stepsize=6,
    shakiness=10,
    accuracy=10,
    result_file="transforms.trf",
):
    """Create a ffmpeg command which uses vidstabdetect.

    Notes:
        https://ffmpeg.org/ffmpeg-filters.html#vidstabdetect-1
    """
    sep = os.sep
    input_path = f"{images_folder}{sep}{file_pattern}".replace("\\", "/")
    result_path = f"{images_folder}/{result_file}".replace("\\", "/")
    escaped_result_path = result_path.replace(":", r"\\:")

    cmd = (
        ffmpeg_cmd + f' -i "{input_path}" -vf '
        f'"vidstabdetect=stepsize={stepsize}:shakiness={shakiness}:accuracy={accuracy}:result={escaped_result_path}" '
        f"-f null -"
    )
    logging.info(cmd)
    return cmd


def ffmpeg_remove_motion_from_frames_command(
    images_folder,
    in_file_pattern="f%05d.jpg",
    out_file_pattern="f%05d.jpg",
    smoothing=0,
    crop="black",
    optzoom=0,
    transform_file="transforms.trf",
    unsharp_filter="5:5:0.8:3:3:0.4",
    quality=2,
):
    """Create a ffmpeg command which uses vidstabtransform.

    Notes:
        https://ffmpeg.org/ffmpeg-filters.html#vidstabtransform-1
    """

    def ffmpeg_escape_windows_path_for_filter(path: str) -> str:
        """
        Convert Windows path to FFmpeg-safe format for filter args:
        - Use forward slashes
        - Escape colon after drive letter (e.g., C\:/...)
        """
        path = Path(path).as_posix()  # Converts backslashes to forward slashes
        if path[1:3] == ":/":
            path = (
                path[0] + r"\:" + path[2:]
            )  # Escape the colon after drive letter
        return path

    transform_path = ffmpeg_escape_windows_path_for_filter(
        f"{images_folder}/{transform_file}"
    )

    cmd = (
        ffmpeg_cmd + f' -i "{images_folder}{os.sep}{in_file_pattern}" '
        f'-vf "vidstabtransform=smoothing={smoothing}:crop={crop}:optzoom={optzoom}:'
        f"input='{transform_path}',"
        f'unsharp={unsharp_filter}" -q:v {quality} "{images_folder}{os.sep}{out_file_pattern}"'
    )

    logging.info(cmd)
    return cmd


def create_ffmpeg_command(params: dict) -> str:
    """Create a valid ffmpeg command including all ffmpeg_process chains requested."""
    command_parts = []

    # Input parameters
    if "start_time" in params:
        command_parts.extend(["-ss", params["start_time"]])
    if "end_time" in params and params["end_time"] is not None:
        command_parts.extend(["-to", params["end_time"]])
    if "input_video" in params:
        command_parts.extend(["-i", quotify_a_string(params["input_video"])])

    # Filters
    filtergraphs = []
    if "video_rotation" in params and params["video_rotation"] > 0:
        filtergraphs.append(f"rotate={params['video_rotation']}*(PI/180)")
    if "video_flip" in params and params["video_flip"] != "none":
        filtergraphs.append(params["video_flip"])
    if "normalize_luma" in params and params["normalize_luma"]:
        filtergraphs.append(
            "scale=out_range=full"
        )  # must also pair with -color_range and -pix-fmt
    if "curve_preset" in params and params["curve_preset"] != "none":
        filtergraphs.append(f"curves=preset={params['curve_preset']}")
    if "extract_frames" in params and params["extract_frames"]:
        filtergraphs.append(
            f"select=not(mod(n\\,{params['extract_frame_step']}))"
        )
    if "calibrate_radial" in params:
        cx, cy, k1, k2 = params["cx"], params["cy"], params["k1"], params["k2"]
        filtergraphs.append(
            f"lenscorrection=cx={cx}:cy={cy}:k1={k1}:k2={k2}:i=bilinear"
        )
    # Format output as 8-bit grayscale (confirmed this give 256 unique colors)
    # filtergraphs.append(f"format=gray")

    # Add filtergraph to command
    if filtergraphs:
        vfilter = ", ".join(filtergraphs)
        command_parts.extend(["-vf", f'"{vfilter}"'])
    else:
        command_parts.extend(
            ["-c:v", "copy"]
        )  # Use stream copy b/c it's fastest

    # Audio and output parameters
    if "strip_audio" in params and "extract_frames" not in params:
        command_parts.extend(["-an"])  # ensure audio gets copied over

    if "extract_frames" in params:
        if params["extract_frames"]:
            output_folder = params["extracted_frames_folder"]
            output_pattern = params["extract_frame_pattern"]
            output_file = os.path.join(output_folder, output_pattern)
            command_parts.extend(
                [
                    "-vsync",
                    "vfr",
                    "-qmin",
                    "1",
                    "-qmax",
                    "1",
                    "-q:v",
                    "1",
                    "-y",
                    quotify_a_string(output_file),
                ]
            )
    if "output_video" in params:
        if params["output_video"][0] != "null -":
            command_parts.extend(
                ["-y", quotify_a_string(params["output_video"])]
            )

    # Construct and return final command string
    command_parts.insert(0, ffmpeg_cmd)
    command = " ".join(command_parts)

    return command


def parse_ffmpeg_stdout_progress(stdout_line_of_text, video_duration=1000):
    """Parse the ffmpeg standard out message and return progress as a percentage of video duration.

    Notes
    -----
    As ffmpeg runs, it generates stdout messages. Many of the messages contain metadata or
    other information not needed to generate a progress indicator. But there are two messages
    which do help. THhe first is in the first stdout text line which gives the input metadata:
        '>>> Duration: 00:00:30.03, start: 0.120000, bitrate: 44868 kb/s
    The second is a line given as ffmpeg process, which contains the currently processing
    frame information:
        '>>> frame=  717 fps=0.0 q=-1.0 Lsize=  164450kB time=00:00:29.86 bitrate=45111.6kbits/s speed= 152x

    This function computes a percentage (from 0 to 100) of the progress given a duration and the
    processing frame time given by ffmpeg.
    """

    time_re_pattern = re.compile(r"time=(\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)")
    matcher = time_re_pattern.search(stdout_line_of_text)

    if matcher:
        time_string = matcher.group(1).replace(",", ".")
        try:
            time_seconds = hhmmss_to_seconds(time_string)
            return int(time_seconds / video_duration * 100)
        except Exception:
            return None  # handle any conversion errors gracefully
    else:
        return None


def ffprobe_add_exif_metadata(file_path, res_dict, ffprobe_cmd="ffprobe"):
    """Add available EXIF-like metadata from ffprobe to the result dictionary.

    Parameters
    ----------
    file_path : str
        Path to the video file.

    res_dict : dict
        Dictionary to update with EXIF metadata (keys prefixed with 'exif_').

    ffprobe_cmd : str
        Path to the ffprobe executable (defaults to 'ffprobe').

    Returns
    -------
    dict
        Updated result dictionary with added EXIF metadata, if available.
    """

    if not os.path.exists(file_path):
        logging.warning(
            f"[ffprobe_add_exif_metadata] File does not exist: {file_path}"
        )
        return res_dict

    cmd = [
        ffprobe_cmd,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        ffprobe_data = json.loads(result.stdout)

        if "format" in ffprobe_data:
            fmt = ffprobe_data["format"]

            # Add tags as exif_* keys
            tags = fmt.get("tags", {})
            for k, v in tags.items():
                res_dict[f"exif_{k.lower()}"] = v

        # Optionally extract pixel format and codec from video stream
        for stream in ffprobe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                if "pix_fmt" in stream:
                    res_dict["pixel_fmt"] = stream["pix_fmt"]
                if "codec_name" in stream:
                    res_dict["codec_info"] = stream["codec_name"]
                break

    except Exception as e:
        logging.warning(
            f"[ffprobe_add_exif_metadata] ffprobe failed for {file_path} with error: {e}"
        )

    return res_dict
