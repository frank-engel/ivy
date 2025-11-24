# IVyTools Linux Installation Guide

This guide covers installation of IVyTools on Linux systems, including Ubuntu, Debian, and Raspberry Pi OS.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [System Dependencies](#system-dependencies)
- [Installation Methods](#installation-methods)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+, Debian 11+, or Raspberry Pi OS (64-bit recommended)
- **Python**: 3.11 or later
- **RAM**: 2GB minimum (4GB recommended for 1080p videos)
- **Disk**: 2GB for installation + space for videos and results
- **CPU**: x86_64 or ARM64 (Raspberry Pi 4/5)

### Recommended Requirements
- **RAM**: 8GB for processing 4K videos
- **SSD**: For faster video I/O (especially on Raspberry Pi)
- **CPU**: Multi-core processor for faster processing

---

## Quick Start

For headless API and batch processing (no GUI):

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv ffmpeg libavcodec-extra

# Create virtual environment
python3 -m venv ivytools-env
source ivytools-env/bin/activate

# Install IVyTools (headless API)
pip install image_velocimetry_tools

# Verify installation
python -c "from image_velocimetry_tools.api import run_batch_processing; print('Success!')"
```

---

## System Dependencies

### Debian/Ubuntu/Raspberry Pi OS

```bash
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    libavcodec-extra \
    libopencv-dev \
    build-essential \
    git
```

**What each package does:**
- `python3`, `python3-pip`, `python3-venv`: Python runtime and package management
- `python3-dev`: Python headers for compiling extensions
- `ffmpeg`, `libavcodec-extra`: Video processing with codec support
- `libopencv-dev`: OpenCV computer vision library
- `build-essential`: Compilers for building Python packages
- `git`: Version control (needed for areacomp3 dependency)

### Fedora/RHEL/CentOS

```bash
sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    ffmpeg \
    ffmpeg-libs \
    opencv \
    opencv-devel \
    gcc \
    gcc-c++ \
    git
```

### Arch Linux

```bash
sudo pacman -S \
    python \
    python-pip \
    ffmpeg \
    opencv \
    base-devel \
    git
```

---

## Installation Methods

### Option 1: pip Install (Recommended)

This is the simplest method for headless API and batch processing.

```bash
# Create and activate virtual environment
python3 -m venv ivytools-env
source ivytools-env/bin/activate

# Install IVyTools
pip install image_velocimetry_tools

# For GUI support (optional):
# pip install image_velocimetry_tools[gui]
```

**Activate environment in future sessions:**
```bash
source ivytools-env/bin/activate
```

**Deactivate when done:**
```bash
deactivate
```

---

### Option 2: Install from Source

For development or latest unreleased features:

```bash
# Clone repository
git clone https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy.git
cd ivy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Or with GUI support:
# pip install -e .[gui]
```

---

### Option 3: Using requirements Files

For specific platform optimizations:

```bash
# For standard Linux
pip install -r requirements-linux.txt

# For Raspberry Pi (optimized)
pip install -r requirements-rpi.txt
```

---

## Verification

### Test Basic Import

```bash
python -c "import image_velocimetry_tools; print(image_velocimetry_tools.__version__)"
```

### Test API Import

```python
from image_velocimetry_tools.api import run_batch_processing
from image_velocimetry_tools.utils.platform import is_linux, is_raspberry_pi

print(f"Running on Linux: {is_linux()}")
print(f"Running on Raspberry Pi: {is_raspberry_pi()}")
```

### Test FFmpeg

```bash
ffmpeg -version
ffprobe -version
```

### Test OpenCV Video Codecs

```python
import cv2
print(f"OpenCV version: {cv2.__version__}")
print(f"Video I/O support: {cv2.getBuildInformation()}")
```

---

## Troubleshooting

### Issue: ModuleNotFoundError for image_velocimetry_tools

**Solution:**
```bash
# Ensure virtual environment is activated
source ivytools-env/bin/activate

# Reinstall package
pip install --upgrade image_velocimetry_tools
```

---

### Issue: FFmpeg not found

**Error:** `FileNotFoundError: FFmpeg and FFprobe not found...`

**Solution:**
```bash
# Install FFmpeg
sudo apt-get install ffmpeg

# Verify installation
which ffmpeg
ffmpeg -version

# Or set environment variable to custom location
export FFMPEG-IVyTools=/path/to/ffmpeg
export FFPROBE-IVyTools=/path/to/ffprobe
```

---

### Issue: OpenCV video fails to open

**Error:** `Unable to open video file`

**Solutions:**

1. **Install additional codecs:**
```bash
sudo apt-get install libavcodec-extra libavformat-dev libavdevice-dev
```

2. **Verify video codec:**
```bash
ffprobe your_video.mp4
```

3. **Convert video to compatible format:**
```bash
ffmpeg -i input.mov -c:v libx264 -crc:a aac output.mp4
```

---

### Issue: NumPy/SciPy compilation errors

**Error:** `Failed building wheel for numpy`

**Solution:**
```bash
# Install build dependencies
sudo apt-get install python3-dev libatlas-base-dev gfortran

# Retry installation
pip install --upgrade pip
pip install image_velocimetry_tools
```

---

### Issue: Permission denied errors

**Solution:**
```bash
# Always use virtual environment (don't use sudo with pip)
python3 -m venv ivytools-env
source ivytools-env/bin/activate
pip install image_velocimetry_tools
```

---

### Issue: Slow installation on Raspberry Pi

**This is normal!** Some packages need to compile on ARM64.

**Tips:**
- Use `--prefer-binary` flag: `pip install --prefer-binary image_velocimetry_tools`
- Installation may take 10-30 minutes on RPi 4, faster on RPi 5
- Ensure adequate cooling during compilation
- Consider using pre-built Docker image (if available)

---

## Performance Tuning

### Raspberry Pi Optimizations

1. **Use external SSD for videos:**
```bash
# Mount SSD
sudo mount /dev/sda1 /mnt/videos

# Process videos from SSD
ivytools-batch --video-dir /mnt/videos --output /mnt/results
```

2. **Increase swap space:**
```bash
# For processing large videos on RPi with limited RAM
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **Enable I2C for temperature monitoring:**
```bash
# Monitor CPU temperature during processing
watch -n 2 'vcgencmd measure_temp'
```

---

## Environment Variables

IVyTools respects the following environment variables:

- `IVYTOOLS_CONFIG_DIR`: Override default config directory
- `IVYTOOLS_CACHE_DIR`: Override default cache directory
- `IVYTOOLS_DATA_DIR`: Override default data directory
- `FFMPEG-IVyTools`: Path to ffmpeg binary
- `FFPROBE-IVyTools`: Path to ffprobe binary
- `IVY_ENV`: Set to `development` for dev mode

**Example:**
```bash
export IVYTOOLS_CONFIG_DIR=/opt/ivytools/config
export FFMPEG-IVyTools=/usr/local/bin/ffmpeg
```

---

## Next Steps

- See [API Documentation](api_reference.md) for using IVyTools in your code
- See [Batch Processing Guide](batch_processing.md) for automated workflows
- See [Raspberry Pi Deployment](raspberry_pi_deployment.md) for field deployment
- See [Troubleshooting Guide](troubleshooting_linux.md) for common issues

---

## Getting Help

- **Issues**: https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/issues
- **Documentation**: https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/wikis/home
- **Email**: fengel@usgs.gov
