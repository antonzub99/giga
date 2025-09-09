import json
import shutil
import subprocess
from pathlib import Path


def make_video(
    input_folder: Path,
    output_video: Path,
    file_pattern: str,
    image_format: str = "png",
    framerate: int = 30,
):
    """
    Create a video from images in the input folder.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in the system's PATH.")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is not installed or not in the system's PATH.")

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    first_frame = next(input_folder.iterdir())

    # Use ffprobe to get the resolution of the first frame
    ffprobe_command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(first_frame)]
    result = subprocess.run(ffprobe_command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    probe_data = json.loads(result.stdout)
    width = probe_data["streams"][0]["width"]
    height = probe_data["streams"][0]["height"]

    # Determine scale filter based on odd/even dimensions
    scale_filter = ""
    if width % 2 == 1 or height % 2 == 1:
        scale_filter = '-vf "scale=ceil(iw/2)*2:ceil(ih/2)*2"'

    command = f"ffmpeg -y -framerate {framerate} -i {input_folder}/{file_pattern}.{image_format}"
    if scale_filter:
        command += f" {scale_filter}"
    command += f" -c:v libx264 -pix_fmt yuv420p -r 30 {output_video}"

    subprocess.run(command, shell=True)
