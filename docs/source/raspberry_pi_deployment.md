# Raspberry Pi Deployment Guide

Deploy IVyTools on Raspberry Pi 4/5 for automated field measurements.

## Overview

This guide covers deploying IVyTools as an automated batch processor on Raspberry Pi at remote field sites for continuous streamflow monitoring.

**Use Case:** Raspberry Pi collects videos of a river using a camera, and IVyTools automatically processes them to compute discharge.

---

## Hardware Requirements

### Recommended Setup
- **Raspberry Pi 5** (8GB RAM) - Best performance
- **Alternative:** Raspberry Pi 4 (4GB or 8GB RAM) - Good performance
- **Storage:** 64GB+ microSD card (Class 10) OR external SSD (recommended)
- **Power:** Official Raspberry Pi Power Supply (5V 3A for RPi 4, 5V 5A for RPi 5)
- **Cooling:** Heatsink and fan (required for sustained processing)
- **Camera:** USB webcam or Raspberry Pi Camera Module

### Optional
- **UPS/Battery Backup:** For power reliability at field sites
- **Weatherproof Enclosure:** For outdoor deployment
- **Cellular Modem:** For remote access
- **External USB Drive:** For video backup

---

## Installation

### 1. Prepare Raspberry Pi OS

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg libavcodec-extra libopencv-dev \
    libatlas-base-dev \
    git \
    vim \
    htop
```

### 2. Install IVyTools

```bash
# Create dedicated user (optional but recommended)
sudo useradd -m -s /bin/bash ivytools
sudo usermod -aG video ivytools

# Switch to ivytools user
sudo su - ivytools

# Create virtual environment
python3 -m venv ~/ivytools-env
source ~/ivytools-env/bin/activate

# Install IVyTools
pip install --prefer-binary image_velocimetry_tools

# Verify installation
python -c "from image_velocimetry_tools.utils.platform import is_raspberry_pi; print(f'RPi detected: {is_raspberry_pi()}')"
```

### 3. Create Directory Structure

```bash
# As ivytools user
mkdir -p ~/ivytools/{videos,processed,config,logs,scaffold}

# Directory structure:
# ~/ivytools/
# ├── videos/          # Incoming videos from camera
# ├── processed/       # Processed results
# ├── config/          # Configuration files
# ├── logs/            # Processing logs
# └── scaffold/        # Scaffold .ivy project
```

---

## Configuration

### 1. Prepare Scaffold Project

The scaffold project contains camera calibration and cross-section geometry that will be applied to all videos.

**On your desktop computer:**
1. Open IVyTools GUI
2. Process one representative video completely
3. Save as `scaffold_project.ivy`
4. Copy to Raspberry Pi: `~/ivytools/scaffold/scaffold_project.ivy`

### 2. Create Batch CSV Template

```bash
cat > ~/ivytools/config/batch_template.csv << 'EOF'
video_path,water_surface_elevation,alpha,start_time,end_time
EOF
```

---

## Automated Processing

### Option 1: Systemd Service (Recommended)

Create a systemd service for continuous monitoring:

```bash
# Create service file
sudo tee /etc/systemd/system/ivytools-watcher.service << 'EOF'
[Unit]
Description=IVyTools Video Processor
After=network.target

[Service]
Type=simple
User=ivytools
WorkingDirectory=/home/ivytools/ivytools
ExecStart=/home/ivytools/ivytools-env/bin/python /home/ivytools/ivytools/watch_and_process.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable ivytools-watcher
sudo systemctl start ivytools-watcher

# Check status
sudo systemctl status ivytools-watcher

# View logs
sudo journalctl -u ivytools-watcher -f
```

### Watch and Process Script

Create `/home/ivytools/ivytools/watch_and_process.py`:

```python
#!/usr/bin/env python3
"""
Watch directory for new videos and process them automatically.
"""
import time
import logging
from pathlib import Path
from datetime import datetime
from image_velocimetry_tools.api import run_batch_processing
from image_velocimetry_tools.utils.platform import is_raspberry_pi

# Configuration
VIDEO_DIR = Path.home() / "ivytools" / "videos"
PROCESSED_DIR = Path.home() / "ivytools" / "processed"
SCAFFOLD_PROJECT = Path.home() / "ivytools" / "scaffold" / "scaffold_project.ivy"
LOG_FILE = Path.home() / "ivytools" / "logs" / "processor.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_water_surface_elevation(video_path):
    """
    Extract water surface elevation from video filename or metadata.

    Expected filename format: YYYY-MM-DD_HHMM_WSE_<elevation>.mp4
    Example: 2025-01-15_1430_WSE_125.5.mp4 -> 125.5 feet
    """
    try:
        # Parse from filename
        parts = video_path.stem.split('_')
        for i, part in enumerate(parts):
            if part == 'WSE' and i + 1 < len(parts):
                return float(parts[i + 1])
    except (ValueError, IndexError):
        pass

    # Default elevation if not found in filename
    logger.warning(f"Could not parse WSE from {video_path.name}, using default")
    return 125.0  # Default - adjust for your site

def create_batch_csv(video_path, output_csv):
    """Create batch CSV for single video."""
    wse = get_water_surface_elevation(video_path)

    with open(output_csv, 'w') as f:
        f.write("video_path,water_surface_elevation,alpha\\n")
        f.write(f"{video_path},{wse},0.85\\n")

def process_video(video_path):
    """Process a single video."""
    logger.info(f"Processing: {video_path.name}")

    # Create output directory for this video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROCESSED_DIR / f"{video_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create batch CSV
    batch_csv = output_dir / "batch.csv"
    create_batch_csv(video_path, batch_csv)

    try:
        # Process video
        results = run_batch_processing(
            scaffold_project=str(SCAFFOLD_PROJECT),
            batch_csv=str(batch_csv),
            output_folder=str(output_dir),
            stop_on_error=False,
        )

        logger.info(f"Completed: {results.successful_jobs}/{results.total_jobs} successful")

        # Move processed video to archive
        archive_dir = PROCESSED_DIR / "archive"
        archive_dir.mkdir(exist_ok=True)
        video_path.rename(archive_dir / video_path.name)

        return True

    except Exception as e:
        logger.error(f"Failed to process {video_path.name}: {e}")
        return False

def watch_directory():
    """Watch directory for new videos and process them."""
    logger.info(f"Starting IVyTools watcher on Raspberry Pi: {is_raspberry_pi()}")
    logger.info(f"Watching: {VIDEO_DIR}")
    logger.info(f"Scaffold: {SCAFFOLD_PROJECT}")

    processed_files = set()

    while True:
        try:
            # Find video files
            video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi"))

            for video_file in video_files:
                if video_file not in processed_files:
                    logger.info(f"New video detected: {video_file.name}")

                    # Wait for file to finish writing (check if size stable)
                    prev_size = 0
                    curr_size = video_file.stat().st_size
                    while curr_size != prev_size:
                        time.sleep(5)
                        prev_size = curr_size
                        curr_size = video_file.stat().st_size

                    # Process video
                    if process_video(video_file):
                        processed_files.add(video_file)

            # Sleep before next check
            time.sleep(30)

        except KeyboardInterrupt:
            logger.info("Watcher stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in watcher: {e}")
            time.sleep(60)

if __name__ == "__main__":
    watch_directory()
```

Make it executable:
```bash
chmod +x /home/ivytools/ivytools/watch_and_process.py
```

---

### Option 2: Cron Job

For scheduled processing rather than continuous monitoring:

```bash
# Edit crontab
crontab -e

# Add entry to run every hour
0 * * * * /home/ivytools/ivytools-env/bin/python /home/ivytools/ivytools/batch_process.py >> /home/ivytools/ivytools/logs/cron.log 2>&1
```

---

## Monitoring and Maintenance

### Check Processing Status

```bash
# View service logs
sudo journalctl -u ivytools-watcher -n 100 --no-pager

# Check CPU temperature
vcgencmd measure_temp

# Monitor resource usage
htop

# Check disk space
df -h
```

### Automatic Log Rotation

```bash
# Create logrotate config
sudo tee /etc/logrotate.d/ivytools << 'EOF'
/home/ivytools/ivytools/logs/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

### Automatic Cleanup

Add to crontab to clean old processed videos:

```bash
# Delete processed videos older than 30 days
0 2 * * * find /home/ivytools/ivytools/processed/archive -name "*.mp4" -mtime +30 -delete
```

---

## Performance Optimization

### Raspberry Pi 4/5 Tweaks

```bash
# Increase GPU memory (if using camera module)
# Edit /boot/config.txt
sudo nano /boot/config.txt

# Add/modify:
gpu_mem=256

# Increase swap size for large videos
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Processing Settings

For Raspberry Pi, use reduced quality settings in your scaffold project:
- Lower frame extraction rate (e.g., 2 fps instead of 5 fps)
- Smaller grid size
- Skip non-essential processing steps

---

## Remote Access

### SSH Access

```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Access from remote
ssh ivytools@raspberrypi.local
```

### VPN for Secure Remote Access

Consider using Tailscale or WireGuard for secure remote access to field sites.

---

## Backup Strategy

### Automatic Backup to External Drive

```bash
# Mount external drive
sudo mount /dev/sda1 /mnt/backup

# Backup script
cat > ~/ivytools/backup.sh << 'EOF'
#!/bin/bash
rsync -av --delete \
    /home/ivytools/ivytools/processed/ \
    /mnt/backup/ivytools_processed/
EOF

chmod +x ~/ivytools/backup.sh

# Add to crontab (daily at 3 AM)
0 3 * * * /home/ivytools/ivytools/backup.sh
```

---

## Troubleshooting

### High CPU Temperature

```bash
# Check temperature
vcgencmd measure_temp

# If >70°C, improve cooling:
# - Add heatsink
# - Add fan
# - Reduce processing quality/rate
# - Process during cooler hours
```

### Out of Memory

```bash
# Check memory usage
free -h

# Solutions:
# - Increase swap
# - Reduce grid size in scaffold
# - Process smaller videos
# - Use RPi 5 with 8GB RAM
```

### Slow Processing

**Expected:** RPi 4 processes 1080p video at ~2-5 minutes/video
**Too slow:** Check:
- CPU throttling due to temperature
- SD card vs SSD (SSD much faster)
- Background processes (use `htop`)

---

## Field Deployment Checklist

- [ ] Raspberry Pi configured and tested
- [ ] IVyTools installed and verified
- [ ] Scaffold project uploaded
- [ ] Watch script configured and tested
- [ ] Systemd service enabled
- [ ] Cooling adequate (heatsink + fan)
- [ ] Storage sufficient (recommend 128GB+ SSD)
- [ ] Power supply reliable (UPS recommended)
- [ ] Remote access configured (SSH/VPN)
- [ ] Monitoring alerts configured
- [ ] Backup strategy implemented
- [ ] Documentation on-site

---

## Support

For field deployment assistance:
- Email: fengel@usgs.gov
- Issues: https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/issues
