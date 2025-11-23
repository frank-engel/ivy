import unittest
import os
from image_velocimetry_tools.ffmpeg_tools import *
from image_velocimetry_tools.common_functions import *

# Priority: ENV var > System PATH > local ./bin fallback
ffmpeg_cmd = os.environ.get("FFMPEG-IVyTools") or shutil.which("ffmpeg")
ffprobe_cmd = os.environ.get("FFPROBE-IVyTools") or shutil.which("ffprobe")

if not ffmpeg_cmd or not ffprobe_cmd:
    fallback_path = Path(resource_path("bin"))
    ffmpeg_fallback = fallback_path / "ffmpeg"
    ffprobe_fallback = fallback_path / "ffprobe"

    if ffmpeg_fallback.exists() and ffprobe_fallback.exists():
        ffmpeg_cmd = str(ffmpeg_fallback)
        ffprobe_cmd = str(ffprobe_fallback)
        logging.warning(
            "[ffmpeg] Using local ./bin/ fallback; system binaries not found."
        )
    else:
        raise FileNotFoundError(
            "FFmpeg and FFprobe not found in environment variables, system PATH, or local ./bin/"
        )


class TestCreateFFmpegCommand(unittest.TestCase):
    # def test_create_command_with_no_input(self):
    #     dictionary = {}
    #     result = create_ffmpeg_command(dictionary)
    #     expected = f"{ffmpeg} -y "
    #     self.assertEqual(result, expected)

    def test_create_command_with_basic_input(self):
        dictionary = {"input_video": "input.mp4", "output_video": "output.mp4"}
        result = create_ffmpeg_command(dictionary)
        expected = f"{ffmpeg_cmd} -i input.mp4 -c:v copy -y output.mp4"
        self.assertEqual(result, expected)

    def test_create_command_with_start_time(self):
        dictionary = {
            "input_video": "input.mp4",
            "output_video": "output.mp4",
            "start_time": "00:01:00.0",
        }
        result = create_ffmpeg_command(dictionary)
        expected = (
            f"{ffmpeg_cmd} -ss 00:01:00.0 -i input.mp4 -c:v copy -y output.mp4"
        )
        self.assertEqual(result, expected)

    def test_create_command_with_end_time(self):
        dictionary = {
            "input_video": "input.mp4",
            "output_video": "output.mp4",
            "end_time": "00:02:30.0",
        }
        result = create_ffmpeg_command(dictionary)
        expected = (
            f"{ffmpeg_cmd} -to 00:02:30.0 -i input.mp4 -c:v copy -y output.mp4"
        )
        self.assertEqual(result, expected)

    def test_create_command_with_extract_frames(self):
        dictionary = {
            "input_video": "input.mp4",
            "extract_frames": True,
            "extract_frame_step": 10,
            "extracted_frames_folder": "frames",
            "extract_frame_pattern": "frame%03d.png",
        }
        result = create_ffmpeg_command(dictionary)
        expected = (
            f'{ffmpeg_cmd} -i input.mp4 -vf "select=not(mod(n\\,10))" -vsync vfr -qmin 1 -qmax 1 -q:v 1 -y '
            f"frames\\frame%03d.png"
        )
        self.assertEqual(result, expected)

    def test_create_command_with_multiple_filters(self):
        dictionary = {
            "input_video": "input.mp4",
            "output_video": "output.mp4",
            "video_rotation": 90,
            "video_flip": "hflip",
            "normalize_luma": True,
            "curve_preset": "darker",
            "strip_audio": True,
        }
        result = create_ffmpeg_command(dictionary)
        expected = (
            f'{ffmpeg_cmd} -i input.mp4 -vf "rotate=90*(PI/180), hflip, scale=out_range=full, '
            f'curves=preset=darker" -an -y output.mp4'
        )
        self.assertEqual(result, expected)

    def test_create_command_with_calibrate_radial(self):
        dictionary = {
            "input_video": "input.mp4",
            "output_video": "output.mp4",
            "calibrate_radial": True,
            "cx": 1280,
            "cy": 720,
            "k1": 0.1,
            "k2": 0.2,
        }
        result = create_ffmpeg_command(dictionary)
        expected = (
            f'{ffmpeg_cmd} -i input.mp4 -vf "lenscorrection=cx=1280:cy=720:k1=0.1:k2=0.2:i=bilinear" '
            f"-y output.mp4"
        )
        self.assertEqual(result, expected)


class TestFFmpegFunctions(unittest.TestCase):
    def setUp(self):
        self.images_folder = "test_images"
        self.test_image_path = os.path.join(self.images_folder, "test.jpg")
        self.frames_folder = "test_frames"
        self.in_file_pattern = "f%05d.jpg"
        self.out_file_pattern = "s%05d.jpg"
        self.transform_file = "transforms.trf"

    #     # Create test image and frames folder
    #     os.makedirs(self.images_folder, exist_ok=True)
    #     os.makedirs(self.frames_folder, exist_ok=True)
    #     # with open(self.test_image_path, "wb") as f:
    #     #     f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x04\x00\x00\x00\xe0"
    #     #             b"?\x7f\x05\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x06PLTE\xff\xff\xff\xff\xff\xff"
    #     #             b"\x00\x00\x00\x9dIDATx\x9c\xed\xc1\x01\x00\x00\x00\x01\x00\x00\x05\xf0\xbd\x8d\xb7\x00\x00\x00"
    #     #             b"\x00IEND\xaeB`\x82")
    #
    #     # Create test frames from test image
    #     subprocess.run(f'{ffmpeg_cmd} -i {self.test_image_path} -vf "fps=10" {os.path.join(self.frames_folder, self.in_file_pattern)}', shell=True, check=True)
    #
    # def tearDown(self):
    #     # Delete test image and frames folder
    #     os.remove(self.test_image_path)
    #     for f in os.listdir(self.frames_folder):
    #         os.remove(os.path.join(self.frames_folder, f))
    #     os.rmdir(self.frames_folder)
    #     os.rmdir(self.images_folder)

    def test_ffmpeg_compute_motion_trajectories_from_frames_command(self):
        # for development directory
        current_directory = os.getcwd()

        # Test default parameters
        cmd = ffmpeg_compute_motion_trajectories_from_frames_command(
            self.frames_folder
        )
        expected = (
            f'{ffmpeg_cmd} -i "test_frames/f%05d.jpg" -vf '
            '"vidstabdetect=stepsize=6:shakiness=10:accuracy=10:result=test_frames/transforms.trf" '
            "-f null -"
        )
        self.assertEqual(cmd, expected)

        # Test custom parameters
        cmd = ffmpeg_compute_motion_trajectories_from_frames_command(
            self.frames_folder,
            stepsize=12,
            shakiness=5,
            accuracy=5,
            result_file="test_transforms.trf",
        )
        expected = (
            f'{ffmpeg_cmd} -i "test_frames/f%05d.jpg" -vf '
            '"vidstabdetect=stepsize=12:shakiness=5:accuracy=5:result'
            '=test_frames/test_transforms.trf" '
            "-f null -"
        )
        self.assertEqual(cmd, expected)


if __name__ == "__main__":
    pass
