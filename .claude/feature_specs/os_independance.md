# OS Independence Feature Specification

## Goal
Remove any Windows or OS specific code for the headless API and replace with OS agnostic capabilities so that IVy will run on Linux. Specifically, the batch and/or API should be able to run on a Raspberry Pi 4 or similar.

## User Story
As a hydrographer, I use a pre-configured Raspberry Pi 4 to collect videos of a river at my gage. I currently download these videos and use the GUI of IVyTools to process streamflow with image velocimetry. I want to do this automatically on my Raspberry Pi.

## Background
IVyTools is currently distributed as a Windows-based application with GUI components built on Qt. The application includes both a GUI interface and API/batch processing capabilities. To enable deployment on edge computing devices like Raspberry Pi at remote monitoring sites, the headless API and batch processor must be platform-agnostic.

## Scope

### In Scope
- Headless API functionality
- Batch processing engine
- Core STIV/PIV computational modules
- File I/O operations
- Path handling and directory operations
- Video codec handling and frame extraction
- Configuration file management
- Logging infrastructure
- Command-line interface
- Process spawning and management (if used)
- System resource checking

### Out of Scope (Initial Phase)
- GUI application components
- Qt-dependent widgets and dialogs
- Windows-specific GUI installers
- LAVFilters codec integration for GUI video playback
- Windows registry operations

## Current Windows Dependencies to Address

### 1. Path Handling
**Issue**: Direct use of Windows path separators or Windows-specific path construction
```python
# Windows-specific (problematic)
path = "C:\\data\\videos\\stream.mp4"
config_path = parent_dir + "\\" + "config.ini"
```

**Solution**: Use `pathlib.Path` or `os.path` for all path operations
```python
# Cross-platform
from pathlib import Path
path = Path("data") / "videos" / "stream.mp4"
config_path = parent_dir / "config.ini"
```

**Action Items**:
- [ ] Audit all modules for hardcoded path separators (`\\` or `/`)
- [ ] Replace string-based path operations with `pathlib.Path`
- [ ] Ensure all file operations use cross-platform path handling
- [ ] Test path operations on both Windows and Linux

### 2. File System Case Sensitivity
**Issue**: Windows file systems are case-insensitive; Linux file systems are case-sensitive

**Solution**: 
- Standardize file naming conventions
- Implement case-sensitive file lookups
- Add validation for file existence checks

**Action Items**:
- [ ] Document file naming conventions (e.g., all lowercase for config files)
- [ ] Add warnings for case mismatches in file references
- [ ] Test file operations with various case combinations on Linux

### 3. Line Endings
**Issue**: Windows uses CRLF (`\r\n`), Unix/Linux uses LF (`\n`)

**Solution**: 
- Open text files with `newline=None` to handle both formats
- Use `.gitattributes` for repository consistency
- Configure text output to use system-appropriate line endings

**Action Items**:
- [ ] Add `.gitattributes` with text file rules
- [ ] Review all text file I/O to use `newline=None`
- [ ] Ensure CSV and config file parsers handle both formats

### 4. Video Codec Dependencies
**Issue**: LAVFilters codec is Windows-specific; OpenCV backend varies by platform

**Solution**: 
- Ensure OpenCV uses platform-appropriate video backends
- On Linux: leverage GStreamer, FFmpeg, or V4L
- On Windows: continue using DirectShow/Media Foundation
- Abstract video reading through a platform-agnostic interface

**Action Items**:
- [ ] Test OpenCV video reading on Linux with common codecs (H.264, H.265, MJPEG)
- [ ] Document required Linux packages (e.g., `ffmpeg`, `gstreamer`, `libavcodec`)
- [ ] Create video backend detection utility
- [ ] Add graceful fallback for unsupported codecs
- [ ] Update installation documentation for Linux codec requirements

### 5. Process and System Calls
**Issue**: Use of Windows-specific APIs (if any) like `os.system()` with Windows commands

**Solution**: 
- Use cross-platform alternatives:
  - Replace `os.system()` with `subprocess.run()`
  - Avoid shell-specific commands
  - Use Python libraries instead of system utilities where possible

**Action Items**:
- [ ] Audit for `os.system()`, `subprocess.call()` with shell commands
- [ ] Replace with cross-platform Python implementations
- [ ] Test subprocess operations on both platforms

### 6. Executable Distribution
**Issue**: Windows `.exe` distribution; Linux needs different packaging

**Solution**: 
- Support multiple distribution methods:
  - Python wheel (`.whl`) for pip installation
  - Docker container for containerized deployment
  - System packages (`.deb` for Debian/Raspbian)
  - Source installation with requirements.txt

**Action Items**:
- [ ] Create `setup.py` for pip-installable package
- [ ] Create Dockerfile for containerized API
- [ ] Document Linux installation procedure
- [ ] Create requirements.txt with platform-specific dependencies
- [ ] Test installation on Raspberry Pi OS (Debian-based)

### 7. Configuration File Locations
**Issue**: Windows config paths (`C:\Users\...`, `C:\ProgramData\...`)

**Solution**: 
- Use XDG Base Directory Specification on Linux
- Support environment variable configuration
- Provide sensible defaults for each platform

```python
# Cross-platform config location
import os
from pathlib import Path

def get_config_dir():
    if os.name == 'nt':  # Windows
        return Path(os.environ.get('APPDATA', Path.home())) / 'IVyTools'
    else:  # Linux/Unix
        return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'ivytools'
```

**Action Items**:
- [ ] Implement platform-specific config directory discovery
- [ ] Document environment variables for configuration override
- [ ] Test config loading on both platforms

### 8. Temporary File Handling
**Issue**: Windows temp directory handling differs from Unix

**Solution**: 
- Use `tempfile` module for all temporary files
- Ensure proper cleanup of temp files

**Action Items**:
- [ ] Audit temporary file creation
- [ ] Replace hardcoded temp paths with `tempfile.mkdtemp()`
- [ ] Implement cleanup handlers

### 9. Performance and Resource Constraints
**Issue**: Raspberry Pi 4 has limited resources compared to desktop PC

**Solution**: 
- Implement memory-efficient processing
- Add configuration options for resource limits
- Support batch size configuration
- Enable processing quality/speed tradeoffs

**Action Items**:
- [ ] Profile memory usage during video processing
- [ ] Add configurable batch processing limits
- [ ] Implement streaming/chunked video processing if needed
- [ ] Add resource monitoring and warnings
- [ ] Document recommended Raspberry Pi settings

### 10. Python Environment Management
**Issue**: Different Python versions and package availability

**Solution**: 
- Specify minimum Python version (recommend 3.8+)
- Pin dependency versions in requirements.txt
- Test on target Python versions for Raspberry Pi OS

**Action Items**:
- [ ] Document minimum Python version
- [ ] Create platform-specific requirements files if needed
- [ ] Test on Python 3.9, 3.10, 3.11 (common on RPi)

## Architecture Changes

### Proposed Module Structure
```
ivytools/
├── api/                    # Headless API (platform-agnostic)
│   ├── __init__.py
│   ├── core.py            # Core API functions
│   ├── batch.py           # Batch processor
│   └── server.py          # Optional REST API server
├── compute/               # Computational modules (platform-agnostic)
│   ├── stiv.py            # STIV algorithms
│   ├── piv.py             # PIV algorithms
│   └── area_comp.py       # Area computation
├── io/                    # I/O operations (platform-agnostic)
│   ├── video.py           # Video reading/writing
│   ├── config.py          # Configuration management
│   └── results.py         # Results export
├── utils/                 # Utilities (platform-agnostic)
│   ├── paths.py           # Path handling
│   ├── logging.py         # Logging utilities
│   └── platform.py        # Platform detection and helpers
├── gui/                   # GUI components (optional, Windows-focused)
│   └── ...                # Qt-based GUI
└── tests/                 # Unit tests
    ├── test_api.py
    ├── test_compute.py
    └── test_io.py
```

### Platform Abstraction Layer
Create a `platform.py` module to handle platform-specific operations:

```python
# ivytools/utils/platform.py
import sys
import platform
from pathlib import Path

def get_platform():
    """Return current platform: 'windows', 'linux', 'darwin'"""
    return sys.platform

def is_windows():
    return sys.platform == 'win32'

def is_linux():
    return sys.platform.startswith('linux')

def is_raspberry_pi():
    """Detect if running on Raspberry Pi"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except:
        return False

def get_cache_dir():
    """Get platform-appropriate cache directory"""
    if is_windows():
        return Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'IVyTools' / 'cache'
    else:
        return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'ivytools'

def get_data_dir():
    """Get platform-appropriate data directory"""
    if is_windows():
        return Path(os.environ.get('APPDATA', Path.home())) / 'IVyTools' / 'data'
    else:
        return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'ivytools'
```

## Testing Requirements

### Unit Tests
- [ ] Add unit tests for all path operations
- [ ] Add unit tests for file I/O
- [ ] Add unit tests for video processing
- [ ] Mock platform-specific calls

### Integration Tests
- [ ] Test complete API workflow on Windows
- [ ] Test complete API workflow on Linux (x86_64)
- [ ] Test complete API workflow on Raspberry Pi 4 (ARM)
- [ ] Test batch processing with various video formats
- [ ] Test with various config file formats

### Performance Tests
- [ ] Benchmark processing time on Raspberry Pi 4
- [ ] Measure memory usage on Raspberry Pi 4
- [ ] Test with different video resolutions
- [ ] Test with different frame rates

## Documentation Updates

### Installation Documentation
- [ ] Create Linux installation guide
- [ ] Create Raspberry Pi specific guide
- [ ] Document system dependencies
- [ ] Document Python environment setup
- [ ] Provide Docker deployment instructions

### API Documentation
- [ ] Document headless API usage
- [ ] Provide Python API examples
- [ ] Document batch processing format
- [ ] Document configuration options

### Troubleshooting Guide
- [ ] Common Linux issues
- [ ] Raspberry Pi specific issues
- [ ] Video codec issues
- [ ] Performance optimization tips

## Deployment Options for Raspberry Pi

### Option 1: Native Python Installation
**Pros**: Direct access, easier debugging, full control
**Cons**: Manual dependency management, OS updates may break environment

```bash
# Installation on Raspberry Pi
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
sudo apt-get install ffmpeg libavcodec-extra libopencv-dev

python3 -m venv ivytools-env
source ivytools-env/bin/activate
pip install ivytools[api]
```

### Option 2: Docker Container
**Pros**: Isolated environment, reproducible, easy updates
**Cons**: Slight overhead, requires Docker knowledge

```dockerfile
# Dockerfile for IVyTools API
FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install IVyTools
COPY . /app
WORKDIR /app
RUN pip install -e .[api]

# Run API
CMD ["python", "-m", "ivytools.api.server"]
```

### Option 3: System Package (.deb)
**Pros**: Standard Linux installation, system integration
**Cons**: More complex to maintain, requires packaging expertise

## Migration Path

### Phase 1: Audit and Refactor (Weeks 1-2)
- [ ] Complete audit of Windows-specific code
- [ ] Refactor path handling
- [ ] Refactor file I/O
- [ ] Create platform abstraction layer

### Phase 2: Core Platform Support (Weeks 3-4)
- [ ] Implement cross-platform video handling
- [ ] Update configuration management
- [ ] Update logging infrastructure
- [ ] Create automated tests

### Phase 3: Raspberry Pi Optimization (Weeks 5-6)
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] Resource limit handling
- [ ] Raspberry Pi testing

### Phase 4: Distribution and Documentation (Weeks 7-8)
- [ ] Create pip-installable package
- [ ] Create Docker container
- [ ] Write installation guides
- [ ] Write deployment guides

## Success Criteria

### Functional Requirements
✓ API successfully processes videos on Linux x86_64
✓ API successfully processes videos on Raspberry Pi 4
✓ Batch processor runs headless on both platforms
✓ All core algorithms produce identical results across platforms
✓ Configuration files work identically on both platforms

### Performance Requirements
✓ Raspberry Pi 4 can process 1080p video at ≥2 FPS
✓ Raspberry Pi 4 can complete typical workflow in ≤30 minutes
✓ Memory usage stays within 2GB on Raspberry Pi

### Quality Requirements
✓ 100% of unit tests pass on both platforms
✓ No platform-specific code outside abstraction layer
✓ Code coverage ≥80% for API modules
✓ Documentation complete for Linux/RPi deployment

## Dependencies

### Required Python Packages (Platform Agnostic)
```
# requirements-api.txt
numpy>=1.20.0
opencv-python>=4.5.0  # or opencv-python-headless
scipy>=1.7.0
pandas>=1.3.0
pyyaml>=5.4.0
python-dateutil>=2.8.0
```

### Optional Dependencies
```
# requirements-server.txt (for REST API)
flask>=2.0.0
flask-restful>=0.3.9
gunicorn>=20.1.0  # for production deployment
```

### System Dependencies (Linux)
```
# For Debian/Raspbian
ffmpeg
libavcodec-extra
libopencv-dev
python3-dev
python3-pip
```

## Risk Assessment

### High Risk
- **Video codec compatibility**: Different codecs may behave differently on Linux
  - Mitigation: Extensive testing with common formats, document supported formats
  
- **Performance on Raspberry Pi**: May be too slow for real-time processing
  - Mitigation: Optimization passes, reduced quality modes, async processing

### Medium Risk
- **Dependency installation**: Some packages may be difficult on ARM
  - Mitigation: Provide pre-built wheels, document workarounds
  
- **File system differences**: Case sensitivity may cause issues
  - Mitigation: Comprehensive testing, clear naming conventions

### Low Risk
- **Path handling**: Well-understood with standard Python libraries
  - Mitigation: Use pathlib consistently

## Future Enhancements

### Post-Launch Improvements
- [ ] GPU acceleration support (OpenCV with CUDA/OpenCL)
- [ ] Distributed processing across multiple Raspberry Pis
- [ ] Real-time streaming video analysis
- [ ] Web-based monitoring dashboard
- [ ] Automatic video ingestion from IP cameras
- [ ] Cloud storage integration (S3, Azure Blob)
- [ ] Edge ML optimization for Raspberry Pi 4

## References
- [OpenCV Platform Support](https://docs.opencv.org)
- [Python pathlib Documentation](https://docs.python.org/3/library/pathlib.html)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
- [Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)

## Appendix A: Quick Reference Checklist

Before declaring OS independence complete:

- [ ] No hardcoded Windows paths (e.g., `C:\`, `\\`)
- [ ] All path operations use `pathlib.Path`
- [ ] No use of `os.system()` with shell commands
- [ ] All file I/O handles both CRLF and LF
- [ ] Video processing tested with H.264, H.265, MJPEG
- [ ] Config files load from platform-appropriate locations
- [ ] Temp files use `tempfile` module
- [ ] All subprocess calls are cross-platform
- [ ] Installation documented for Linux
- [ ] Tested on Raspberry Pi 4
- [ ] Docker container builds successfully
- [ ] pip package installs on Linux
- [ ] All tests pass on Windows, Linux, and Raspberry Pi

## Appendix B: Example Batch Processing Script for Raspberry Pi

```python
#!/usr/bin/env python3
"""
batch_process_rpi.py - Example batch processing script for Raspberry Pi
"""
from pathlib import Path
from ivytools.api import IVyAPI
from ivytools.utils.platform import is_raspberry_pi, get_data_dir

def main():
    # Configuration
    video_dir = Path("/home/pi/river_videos")
    output_dir = get_data_dir() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimize for Raspberry Pi
    config = {
        'max_memory_gb': 1.5 if is_raspberry_pi() else 4.0,
        'processing_threads': 2 if is_raspberry_pi() else 4,
        'frame_skip': 2 if is_raspberry_pi() else 1,  # Process every 2nd frame on RPi
    }
    
    # Initialize API
    api = IVyAPI(config)
    
    # Process all videos in directory
    for video_file in video_dir.glob("*.mp4"):
        print(f"Processing {video_file.name}...")
        result = api.process_video(
            video_path=video_file,
            output_path=output_dir / f"{video_file.stem}_result.csv"
        )
        print(f"  Discharge: {result['discharge']:.2f} m³/s")
    
    print("Batch processing complete!")

if __name__ == "__main__":
    main()
```