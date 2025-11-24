# OS Independence Implementation Plan

**Feature:** OS Independence for Headless API and Batch Processing
**Target Platforms:** Linux (Ubuntu/Debian), Raspberry Pi 4 (Raspbian/Raspberry Pi OS)
**Priority:** High
**Status:** In Progress
**Created:** 2025-11-24
**Branch:** `claude/implement-claude-task-01GNXWVPvL7RmtwAnFJ2hSFY`

---

## Executive Summary

Transform IVyTools from a Windows-focused application to a cross-platform solution that runs on Linux and Raspberry Pi 4. Primary focus is on the headless API and batch processor to enable automated river flow monitoring at remote field sites.

**Key Goals:**
- Headless API runs on Raspberry Pi 4
- Batch processor runs on Linux without GUI dependencies
- pip-installable package for easy deployment
- Identical computation results across all platforms
- Performance optimized for resource-constrained devices

---

## Current State Assessment

### ✅ Already Good
- MVP architecture with Qt-free services layer
- `pathlib.Path` used in ~20 files
- `opencv-python-headless` in requirements (headless-friendly)
- API and batch processing layers already exist
- Python 3.11+ requirement
- `subprocess.run()` used (no `os.system()` calls)

### ❌ Issues to Fix
- **ffmpeg_tools.py:29-46** - Hardcoded `.exe` extensions
- **file_management.py** - Windows-specific filename sanitization
- **requirements.txt** - Contains `pywin32-ctypes` (Windows-only)
- **8 files** - Use backslash path separators (`\\`)
- **No platform abstraction layer**
- **No utils/ directory structure**

---

## Implementation Phases

### Phase 1: Infrastructure & Platform Abstraction (Week 1)

#### 1.1 Create Platform Utilities Module
**Priority:** High | **Effort:** Medium

**Directory Structure:**
```
image_velocimetry_tools/
└── utils/
    ├── __init__.py
    ├── platform.py      # Platform detection & helpers
    ├── paths.py         # Path handling utilities
    └── filesystem.py    # File operations
```

**`utils/platform.py` Functions:**
```python
- get_platform() -> str                    # Return 'windows', 'linux', 'darwin'
- is_windows() -> bool
- is_linux() -> bool
- is_mac() -> bool
- is_raspberry_pi() -> bool               # Detect RPi hardware via /proc/device-tree/model
- get_config_dir() -> Path                # XDG spec on Linux, AppData on Windows
- get_cache_dir() -> Path                 # Platform-appropriate cache location
- get_data_dir() -> Path                  # Platform-appropriate data location
```

**`utils/filesystem.py` Functions:**
```python
- make_safe_filename(input_string: str) -> str  # Cross-platform safe filenames
- ensure_directory(path: Path) -> None          # Create with proper permissions
- get_file_size(path: Path) -> int              # Cross-platform file size
```

**Tests:**
- `tests/test_utils/test_platform.py`
- `tests/test_utils/test_filesystem.py`

---

#### 1.2 Fix FFmpeg Binary Detection
**Priority:** High | **Effort:** Low

**File:** `image_velocimetry_tools/ffmpeg_tools.py`

**Change:**
```python
# Before (lines 29-30, 41-42)
ffmpeg_fallback = fallback_path / "ffmpeg.exe"
ffprobe_fallback = fallback_path / "ffprobe.exe"

# After
from image_velocimetry_tools.utils.platform import is_windows
exe_ext = ".exe" if is_windows() else ""
ffmpeg_fallback = fallback_path / f"ffmpeg{exe_ext}"
ffprobe_fallback = fallback_path / f"ffprobe{exe_ext}"
```

**Tests:** Update `tests/test_ffmpeg_tools.py` to test on both platforms

---

#### 1.3 Replace Windows-Specific Path Handling
**Priority:** High | **Effort:** Medium

**Files to Audit (8 files with `\\`):**
- `image_velocimetry_tools/file_management.py`
- `image_velocimetry_tools/ffmpeg_tools.py`
- `image_velocimetry_tools/gui/controllers/ortho_controller.py`
- `image_velocimetry_tools/gui/HomographyDistanceConversionTool.py`
- `image_velocimetry_tools/services/job_executor.py`
- `tests/test_services/test_project_service.py`
- `tests/test_ffmpeg_tools.py`
- `tests/test_file_management.py`

**Pattern to Fix:**
```python
# Before
path = base_dir + "\\" + "results" + "\\" + filename

# After
from pathlib import Path
path = Path(base_dir) / "results" / filename
```

**Replace `make_windows_safe_filename()`:**
- Move to `utils/filesystem.py` as `make_safe_filename()`
- Handle Windows, Linux, macOS restrictions
- Update all callers

---

#### 1.4 Handle Line Endings
**Priority:** Medium | **Effort:** Low

**Create `.gitattributes`:**
```gitattributes
* text=auto
*.py text eol=lf
*.md text eol=lf
*.csv text eol=lf
*.json text eol=lf
*.txt text eol=lf
*.sh text eol=lf
*.bat text eol=crlf
```

**Verify:** All text file I/O uses `newline=None` (Python default handles both formats)

---

### Phase 2: Dependencies & Requirements (Week 2)

#### 2.1 Update Requirements Files
**Priority:** High | **Effort:** Low

**Issues:**
- `pywin32-ctypes==0.2.3` is Windows-only (remove from cross-platform deps)
- `requirements.txt` appears UTF-16 encoded (fix to UTF-8)

**New Structure:**
```
requirements-base.txt         # Core deps (all platforms)
requirements-linux.txt        # Linux-specific (references base)
requirements-rpi.txt          # RPi optimizations (references linux)
requirements-dev.txt          # Development tools
requirements.txt              # Full frozen requirements (development)
install_requires.txt          # Relaxed constraints (distribution)
```

**Update `setup.py`:**
```python
extras_require={
    'api': [],  # Headless API dependencies (no PyQt)
    'gui': ['PyQt5>=5.15.11'],  # GUI-only
    'dev': ['pytest', 'black', 'pylint'],
}
```

**Remove from base requirements:**
- `pywin32-ctypes` (Windows-only)

---

#### 2.2 Document Linux System Dependencies
**Priority:** High | **Effort:** Low

**Create:** `docs/source/installation_linux.md`

**Sections:**
1. System Requirements (Python, RAM, disk)
2. System Dependencies (apt/dnf commands)
3. Installation Options:
   - pip install (headless API)
   - From source
   - Docker (optional)
4. Raspberry Pi Specific Instructions
5. Verification Steps
6. Troubleshooting

**Key Content:**
```bash
# Debian/Ubuntu/Raspberry Pi OS
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg libavcodec-extra \
    libopencv-dev \
    python3-dev

# Install IVyTools headless API
python3 -m venv ivytools-env
source ivytools-env/bin/activate
pip install image_velocimetry_tools[api]
```

---

#### 2.3 Test OpenCV Video Codecs (Optional - Week 2 if time allows)
**Priority:** Medium | **Effort:** Medium

Create test script to verify H.264, H.265, MJPEG support on Linux.

---

### Phase 3: Core Module Updates (Week 3)

#### 3.1 Update File Management Module
**Files:** `image_velocimetry_tools/file_management.py`

- Replace `make_windows_safe_filename()` with cross-platform `make_safe_filename()`
- Update all callers
- Handle null bytes, control chars, platform-specific restrictions

---

#### 3.2 Configuration File Management
**Create:** `utils/config.py`

Platform-specific config directories:
- Windows: `%APPDATA%/IVyTools`
- Linux: `~/.config/ivytools` (XDG spec)
- Environment override: `IVYTOOLS_CONFIG_DIR`

---

### Phase 4: Testing & Validation (Week 4)

#### 4.1 Cross-Platform Test Suite
- Add pytest markers: `@pytest.mark.linux`, `@pytest.mark.windows`, `@pytest.mark.rpi`
- Create integration tests for platform independence
- Set up CI/CD matrix (GitHub Actions) for Ubuntu, Windows, macOS

#### 4.2 Raspberry Pi Performance Testing
- Benchmark video processing on RPi 4
- Test with 720p, 1080p videos
- Monitor memory usage (target: ≤2GB)
- Document performance characteristics

---

### Phase 5: Distribution & Deployment (Week 5)

#### 5.1 pip-Installable Package (Primary)
**Priority:** High

**Update `setup.py`:**
```python
entry_points={
    'console_scripts': [
        'ivytools-batch=image_velocimetry_tools.api.batch_api:main',
    ],
},
classifiers=[
    'Operating System :: POSIX :: Linux',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS',
],
```

**Create:** `pyproject.toml` (modern Python packaging)

---

#### 5.2 Docker Container (Optional, if useful)
**Priority:** Medium

Multi-stage Dockerfile with ARM64 support for Raspberry Pi.

**Note:** Only pursue if it simplifies deployment alongside existing video collection app.

---

#### 5.3 Raspberry Pi Deployment
**Create:**
- `scripts/install_rpi.sh` - Installation script
- `scripts/ivytools.service` - systemd service for auto-start
- `docs/source/raspberry_pi_deployment.md` - Deployment guide

---

### Phase 6: Documentation (Week 6)

**Create/Update:**
1. `docs/source/installation_linux.md` - Linux installation
2. `docs/source/installation_raspberry_pi.md` - RPi-specific guide
3. `docs/source/api_headless.md` - Headless API usage
4. `docs/source/batch_processing_cli.md` - CLI batch processing
5. `docs/source/troubleshooting_linux.md` - Common issues
6. Update `README.md` - Remove Windows-only references

---

### Phase 7: Final Validation (Week 7)

**Functional Testing:**
- [ ] API processes videos on Ubuntu 22.04 LTS
- [ ] API processes videos on Raspberry Pi 4
- [ ] Batch processor runs headless on both platforms
- [ ] Identical results across Windows, Linux, macOS
- [ ] pip package installs on Linux
- [ ] Config files work on all platforms

**Performance Testing:**
- [ ] RPi 4 processes 1080p video (acceptable rate)
- [ ] Memory usage ≤2GB on RPi
- [ ] No memory leaks during batch runs

**Quality Testing:**
- [ ] All unit tests pass on Windows, Linux, macOS
- [ ] Integration tests pass on all platforms
- [ ] Code coverage ≥80% for utils modules
- [ ] No platform-specific code outside utils/

---

## Success Criteria

### Must Have (Headless API Focus)
✅ API successfully processes videos on Ubuntu 22.04 LTS
✅ API successfully processes videos on Raspberry Pi 4
✅ Batch processor runs headless on both platforms
✅ Identical results across Windows, Linux, macOS
✅ pip installation works: `pip install image_velocimetry_tools[api]`
✅ Documentation complete for Linux/RPi headless deployment
✅ No GUI dependencies required for headless API

### Should Have
✅ Processing time on RPi ≤ 2x desktop Linux
✅ Memory usage on RPi ≤ 2GB
✅ Systemd service for auto-start
✅ CI/CD testing on Ubuntu, Windows, macOS

### Nice to Have
- Docker deployment option (if it simplifies integration)
- ARM64 optimizations
- Performance monitoring tools
- Automatic video ingestion integration

---

## Risk Mitigation

### High Risk: Video Codec Compatibility
**Risk:** OpenCV may not support all codecs on Linux
**Mitigation:** Test H.264, H.265, MJPEG extensively on Linux
**Fallback:** Document supported codecs, provide conversion tools

### High Risk: Performance on Raspberry Pi
**Risk:** RPi 4 may be too slow for large videos
**Mitigation:** Implement configurable quality/speed tradeoffs
**Fallback:** Recommend RPi 5 or recommend desktop Linux for 4K videos

### Medium Risk: Dependency Installation on ARM
**Risk:** Some Python packages may not have ARM64 wheels
**Mitigation:** Test all dependencies on RPi 4, document build process
**Fallback:** Provide Docker image with pre-built dependencies

### Low Risk: Path Handling Edge Cases
**Risk:** Missed path separators in obscure code paths
**Mitigation:** Comprehensive grep audit and testing
**Fallback:** pathlib is well-tested, edge cases unlikely

---

## Notes

### Headless Focus
- GUI components (PyQt5) are NOT required for headless API
- Only include GUI dependencies in `extras_require['gui']`
- All core computation in services layer (Qt-free)

### Integration with Video Collection App
- pip install approach allows side-by-side installation
- Can be imported as Python module: `from image_velocimetry_tools.api import run_batch_processing`
- Systemd service can watch directory for new videos

### Raspberry Pi Considerations
- Target: Raspberry Pi 4 (4GB or 8GB RAM recommended)
- OS: Raspberry Pi OS (64-bit recommended)
- Processing: Expect 2-10 minutes per video depending on length/resolution
- Storage: Recommend external SSD for video storage (faster I/O)

---

## References

- Feature Spec: `.claude/feature_specs/os_independance.md`
- Architecture: `.claude/architechture.md`
- Coding Standards: `.claude/coding_standards.md`
- Workflow: `.claude/workflow.md`
- Python pathlib: https://docs.python.org/3/library/pathlib.html
- XDG Base Directory: https://specifications.freedesktop.org/basedir-spec/
- OpenCV Platform Support: https://docs.opencv.org
