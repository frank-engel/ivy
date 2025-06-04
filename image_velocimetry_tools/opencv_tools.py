"""IVy module that manages openCV functions"""
import logging

import cv2
import os
from image_velocimetry_tools.ffmpeg_tools import ffprobe_frame_count


def opencv_get_video_metadata(file_path, status_callback=None):
    """Extract video metadata using OpenCV with frame count fallback."""
    res = {
        "width": -1,
        "height": -1,
        "bitrate_kbs": -999,
        "duration": -1,
        "avg_frame_rate": -1,
        "avg_timestep_ms": -1,
        "frame_count": -1,
        "codec_info": "unknown",
        "pixel_fmt": "-999",
    }

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return res

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {file_path}")
        return res

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # If OpenCV frame count is invalid, fallback
    if frame_count <= 0 or frame_count > 1e6:
        msg = ("[Warning] OpenCV frame count invalid, "
               "using slower fallback. Press OK to attempt "
               "parsing the video.")
        logging.warning(msg)
        if status_callback:
            status_callback(msg)
        frame_count = count_frames_manual(file_path)

        if frame_count is None:
            msg = ("[Warning] OpenCV manual frame count fallback failed, "
                   "counting frames using full ffprobe decode (SLOWEST). "
                   "Press OK to attempt parsing the video.")
            logging.warning(msg)
            if status_callback:
                status_callback(msg)
            frame_count = ffprobe_frame_count(file_path)

    duration = frame_count / fps if fps > 0 else -1

    # Codec info
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    res.update({
        "width": width,
        "height": height,
        "bitrate_kbs": -999,  # OpenCV limitation
        "duration": duration,
        "avg_frame_rate": fps,
        "avg_timestep_ms": 1 / fps * 1000 if fps > 0 else -1,
        "frame_count": int(frame_count),
        "codec_info": codec,
        "pixel_fmt": "-999",  # OpenCV limitation
    })

    return res


def count_frames_manual(file_path):
    """Count frames manually using OpenCV (slow)."""
    cap = cv2.VideoCapture(file_path)
    total = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        total += 1
    cap.release()
    return total
