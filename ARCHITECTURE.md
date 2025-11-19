# IVyTools Architecture Documentation

**Version:** 2.0 (Post-Refactoring)
**Last Updated:** 2025-11-19
**Status:** Production

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Pattern](#architecture-pattern)
3. [Directory Structure](#directory-structure)
4. [Component Layers](#component-layers)
5. [Data Flow](#data-flow)
6. [Key Patterns](#key-patterns)
7. [Service Layer](#service-layer)
8. [Controller Layer](#controller-layer)
9. [Model Layer](#model-layer)
10. [Testing Strategy](#testing-strategy)
11. [Adding New Features](#adding-new-features)
12. [Code Examples](#code-examples)
13. [Best Practices](#best-practices)

---

## Overview

IVyTools is a PyQt5-based desktop application for image velocimetry analysis. The application processes river imagery to compute surface velocities and discharge measurements using Space-Time Image Velocimetry (STIV) techniques.

### Core Technologies

- **GUI Framework:** PyQt5
- **Image Processing:** OpenCV, PIL, scikit-image
- **Scientific Computing:** NumPy, SciPy, pandas
- **Geospatial:** AreaComp3 (cross-section analysis)
- **Video Processing:** FFmpeg
- **Testing:** pytest

### Architecture Philosophy

The architecture follows the **Model-View-Presenter (MVP)** pattern with an additional **Service Layer** for business logic:

- **Separation of Concerns:** UI, business logic, and state are cleanly separated
- **Testability:** Business logic is 100% testable without GUI
- **Maintainability:** Components are focused and modular
- **Reusability:** Services can be used from CLI, API, or GUI

---

## Architecture Pattern

### MVP with Service Layer

```
┌─────────────────────────────────────────────────────────────┐
│                         View Layer                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  IvyToolsMainWindow (ivy.py)                       │    │
│  │  • Qt widgets and UI components                    │    │
│  │  • Displays data and captures user input           │    │
│  │  • Delegates all logic to Controllers              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↕ (signals/slots)
┌─────────────────────────────────────────────────────────────┐
│                      Presenter Layer                        │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐   │
│  │VideoController│  │OrthoController│  │GridController│   │
│  │               │  │               │  │              │   │
│  │ProjectCtrl    │  │SettingsCtrl   │  │...           │   │
│  └───────────────┘  └───────────────┘  └──────────────┘   │
│  • Orchestrate UI and Services                             │
│  • Handle user interactions                                │
│  • Update UI based on Model changes                        │
│  • NO business logic                                       │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                        Model Layer                          │
│  ┌───────────┐  ┌───────────┐  ┌────────────┐             │
│  │VideoModel │  │OrthoModel │  │GridModel   │             │
│  │           │  │           │  │            │             │
│  │ProjectMdl │  │SettingsMdl│  │...         │             │
│  └───────────┘  └───────────┘  └────────────┘             │
│  • Hold application state                                  │
│  • Emit Qt signals on state changes                        │
│  • Validate state transitions                              │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                       Service Layer                         │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │VideoService    │  │OrthoService  │  │GridService    │  │
│  │ProjectService  │  │ImageStackSvc │  │DischargeSvc   │  │
│  │STIVService     │  │CrossSection  │  │...            │  │
│  └────────────────┘  └──────────────┘  └───────────────┘  │
│  • Pure business logic (no Qt dependencies)                │
│  • Testable without GUI                                    │
│  • Reusable from any interface                             │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                  Core Business Logic Modules                │
│  ┌──────────┐  ┌───────────────┐  ┌─────────────────┐     │
│  │stiv.py   │  │ortho.py       │  │common_funcs.py  │     │
│  │          │  │               │  │                 │     │
│  │image_proc│  │file_mgmt.py   │  │...              │     │
│  └──────────┘  └───────────────┘  └─────────────────┘     │
│  • Low-level algorithms and utilities                      │
│  • Domain-specific calculations                            │
│  • No UI awareness                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
image_velocimetry_tools/
├── gui/
│   ├── ivy.py                       # Main window (5,817 lines)
│   ├── controllers/                 # Presenter layer
│   │   ├── __init__.py
│   │   ├── base_controller.py       # Base controller class
│   │   ├── video_controller.py      # Video playback coordination
│   │   ├── project_controller.py    # Project management
│   │   ├── ortho_controller.py      # Orthorectification UI
│   │   ├── grid_controller.py       # Grid generation UI
│   │   └── settings_controller.py   # Settings/preferences
│   ├── models/                      # Application state
│   │   ├── __init__.py
│   │   ├── base_model.py            # Base model with signals
│   │   ├── video_model.py           # Video state
│   │   ├── project_model.py         # Project state
│   │   ├── ortho_model.py           # Ortho state
│   │   ├── grid_model.py            # Grid state
│   │   └── settings_model.py        # Settings state
│   ├── discharge.py                 # Discharge tab
│   ├── stiv_processor.py            # STIV processing tabs
│   ├── stiv_helper.py               # STIV helper dialog
│   ├── xsgeometry.py                # Cross-section geometry tab
│   └── dialogs/                     # Various dialog windows
│       └── *.py
├── services/                        # Business logic layer
│   ├── __init__.py
│   ├── base_service.py              # Base service class
│   ├── video_service.py             # Video processing logic
│   ├── project_service.py           # Project save/load
│   ├── orthorectification_service.py # Ortho calculations
│   ├── grid_service.py              # Grid generation
│   ├── image_stack_service.py       # Image stack creation
│   ├── discharge_service.py         # Discharge calculations
│   ├── stiv_service.py              # STIV coordination
│   └── cross_section_service.py     # XS geometry calculations
├── stiv.py                          # Core STIV algorithms
├── orthorectification.py            # Core ortho algorithms
├── common_functions.py              # Utility functions
├── image_processing_tools.py        # Image processing
├── file_management.py               # File I/O utilities
├── uncertainty.py                   # Uncertainty analysis
└── tests/
    ├── test_services/               # Service layer tests
    │   ├── test_video_service.py
    │   ├── test_project_service.py
    │   ├── test_orthorectification_service.py
    │   ├── test_grid_service.py
    │   ├── test_discharge_service.py      # 23 tests
    │   ├── test_stiv_service.py           # 23 tests
    │   └── test_cross_section_service.py  # 55+ tests
    └── test_*.py                    # Core module tests
```

---

## Component Layers

### 1. View Layer (Qt Widgets)

**Location:** `gui/ivy.py`, `gui/discharge.py`, `gui/stiv_processor.py`, etc.

**Responsibilities:**
- Display UI components (buttons, tables, plots)
- Capture user input
- Forward events to Controllers
- Update widgets based on Model state

**Key Principles:**
- **No business logic** - all calculations delegated to Services
- **No direct state management** - state held in Models
- **Reactive to Model changes** - subscribes to Model signals

**Example:**
```python
# View listens to Model changes
self.video_model.position_changed.connect(self.update_video_slider)

# View forwards user action to Controller
self.pushButton_play.clicked.connect(self.video_controller.on_play_clicked)
```

---

### 2. Presenter Layer (Controllers)

**Location:** `gui/controllers/*.py`

**Responsibilities:**
- Orchestrate between View, Model, and Service
- Handle UI events
- Update Models based on Service results
- Coordinate complex workflows
- **NO business logic** - delegate to Services

**Key Principles:**
- One controller per feature area
- Reference to main window for widget access
- Use Models for state
- Call Services for business logic

**Example:**
```python
class VideoController(BaseController):
    def __init__(self, main_window, video_model, video_service):
        self.main_window = main_window
        self.video_model = video_model
        self.video_service = video_service

    def on_play_clicked(self):
        """Orchestrate video playback."""
        # Get state from Model
        if self.video_model.is_playing:
            self.video_model.pause()
        else:
            # Call Service for business logic
            can_play = self.video_service.validate_playback(
                self.video_model.video_path
            )
            if can_play:
                self.video_model.play()
```

---

### 3. Model Layer (Application State)

**Location:** `gui/models/*.py`

**Responsibilities:**
- Hold application state
- Emit Qt signals when state changes
- Validate state transitions
- Provide properties for controlled access

**Key Principles:**
- Inherit from `QObject` for signals
- Use properties with getters/setters
- Emit signal when state changes
- No business logic

**Example:**
```python
class VideoModel(QObject):
    position_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int):
        if value != self._position:
            self._position = value
            self.position_changed.emit(value)
```

---

### 4. Service Layer (Business Logic)

**Location:** `services/*.py`

**Responsibilities:**
- Implement business logic
- Perform calculations
- Validate data
- Coordinate with core modules
- **Zero Qt dependencies**

**Key Principles:**
- Stateless or minimal state
- Pure Python/NumPy (no PyQt5 imports)
- Testable without GUI
- Type hints for clarity
- Well-documented

**Example:**
```python
class VideoService:
    def build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float
    ) -> List[str]:
        """Build FFmpeg command for video clip extraction."""
        self._validate_clip_times(start_time, end_time)
        # Business logic here
        return command
```

---

### 5. Core Module Layer

**Location:** Root level (`stiv.py`, `orthorectification.py`, etc.)

**Responsibilities:**
- Low-level algorithms
- Domain-specific calculations
- Utility functions
- No UI awareness

**Key Principles:**
- Pure functions where possible
- NumPy/SciPy-based implementations
- Well-tested
- Reusable across the application

---

## Data Flow

### Typical User Interaction Flow

```
┌─────────────┐
│    User     │ 1. Clicks "Load Video" button
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│  View (ivy.py)  │ 2. Emits clicked signal
└──────┬──────────┘
       │
       ↓
┌──────────────────┐
│ VideoController  │ 3. Calls on_load_video()
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  VideoService    │ 4. Validates video path
└──────┬───────────┘    Extracts metadata
       │
       ↓
┌──────────────────┐
│   VideoModel     │ 5. Updates state
└──────┬───────────┘    Emits video_loaded signal
       │
       ↓
┌─────────────────┐
│  View (ivy.py)  │ 6. Receives signal
└──────┬──────────┘    Updates UI widgets
       │
       ↓
┌─────────────┐
│    User     │ 7. Sees updated UI
└─────────────┘
```

### Signal Flow Diagram

```
View ──clicked──> Controller ──call──> Service
                      │                   │
                      │                   │
                   update               return
                      │                   │
                      ↓                   ↓
                   Model ──signal──> View updates
```

---

## Key Patterns

### 1. Dependency Injection

Controllers and services are injected into components:

```python
class IvyToolsMainWindow:
    def __init__(self):
        # Create services
        self.video_service = VideoService()
        self.project_service = ProjectService()

        # Create models
        self.video_model = VideoModel()
        self.project_model = ProjectModel()

        # Create controllers (inject dependencies)
        self.video_controller = VideoController(
            self, self.video_model, self.video_service
        )
        self.project_controller = ProjectController(
            self, self.project_model, self.project_service
        )
```

### 2. Delegation Pattern

GUI methods delegate to controllers/services:

```python
# In ivy.py
def on_load_video_clicked(self):
    """Delegate to controller."""
    self.video_controller.load_video()

# In VideoController
def load_video(self):
    """Orchestrate video loading."""
    path = self._show_file_dialog()
    if path:
        # Delegate to service
        metadata = self.video_service.get_metadata(path)
        # Update model
        self.video_model.video_path = path
        self.video_model.duration = metadata['duration']
```

### 3. Observer Pattern (Qt Signals)

Models emit signals, Views observe:

```python
# Model emits
class VideoModel(QObject):
    position_changed = pyqtSignal(int)

    @position.setter
    def position(self, value: int):
        self._position = value
        self.position_changed.emit(value)  # Notify observers

# View observes
class IvyToolsMainWindow:
    def __init__(self):
        self.video_model.position_changed.connect(
            self.on_video_position_changed
        )

    def on_video_position_changed(self, position: int):
        """Update slider when position changes."""
        self.slider_video.setValue(position)
```

### 4. Service Locator (Simple)

Services are stored as instance variables:

```python
# Access services via self
result = self.video_service.process_frame(frame)
data = self.project_service.load_project(path)
```

---

## Service Layer

### Service Design Principles

1. **Stateless or Minimal State**
   - Services should not hold mutable state
   - Configuration can be passed to `__init__`
   - Most methods should be independent

2. **No Qt Dependencies**
   - Import only Python stdlib, NumPy, SciPy, pandas
   - Never import from PyQt5
   - Return Python native types

3. **Single Responsibility**
   - Each service handles one feature area
   - Keep services focused (200-500 lines typical)

4. **Testable**
   - All methods have clear inputs/outputs
   - No global state dependencies
   - Easy to mock dependencies

### Service Examples

#### VideoService (376 lines)

**Purpose:** Video processing and FFmpeg operations

**Key Methods:**
- `build_ffmpeg_command()` - Generate FFmpeg commands
- `get_video_metadata()` - Extract video properties
- `validate_clip_times()` - Validate time ranges
- `extract_frames()` - Coordinate frame extraction

**Dependencies:** FFmpeg (subprocess), OpenCV

---

#### DischargeService (346 lines)

**Purpose:** Discharge calculations and uncertainty analysis

**Key Methods:**
- `get_station_and_depth()` - Extract cross-section data
- `create_discharge_dataframe()` - Build calculation dataframe
- `compute_discharge()` - Mid-section method
- `compute_uncertainty()` - ISO 748 / IVE uncertainty
- `compute_summary_statistics()` - Flow statistics

**Dependencies:** NumPy, pandas, Uncertainty class

**Test Coverage:** 23 test cases, 100% of business logic

---

#### STIVService (373 lines)

**Purpose:** STIV-related calculations and data processing

**Key Methods:**
- `compute_sti_velocity()` - Fujita et al. (2007) equation
- `compute_sti_angle()` - Inverse calculation
- `compute_velocity_from_manual_angle()` - Manual processing
- `load_stiv_results_from_csv()` - Data loading
- `apply_manual_corrections()` - Apply edits
- `compute_optimum_sample_time()` - Optimization

**Dependencies:** NumPy, pandas, STIV core module

**Test Coverage:** 23 test cases

---

#### CrossSectionService (425 lines)

**Purpose:** Cross-section geometry calculations

**Key Methods:**
- `compute_pixel_distance()` - Euclidean distance
- `find_station_crossings()` - Water surface crossings
- `interpolate_elevations()` - Linear interpolation
- `check_duplicate_stations()` - Data validation
- `compute_wetted_width()` - Wetted width calculation
- `compute_channel_area()` - Trapezoidal integration
- `flip_stations()` - Bank orientation reversal

**Dependencies:** NumPy, pandas

**Test Coverage:** 55+ test cases (most comprehensive!)

---

## Controller Layer

### Controller Design Principles

1. **UI Orchestration Only**
   - Coordinate between View, Model, Service
   - Handle Qt signals and slots
   - Update Models, not widgets directly
   - Never perform calculations

2. **Reference to Main Window**
   - Access widgets via `self.main_window`
   - Can show dialogs
   - Can update status bar

3. **Use Models for State**
   - Read state from Models
   - Update Models, which emit signals
   - Let View react to Model signals

4. **Delegate to Services**
   - Call services for all business logic
   - Pass service results to Models

### Controller Examples

#### VideoController (654 lines)

**Purpose:** Coordinate video playback and processing UI

**Responsibilities:**
- Video playback control (play/pause/seek)
- FFmpeg thread management
- Frame extraction coordination
- Video timeline updates

**Model:** VideoModel (path, position, duration, is_playing, clip times)

**Service:** VideoService

---

#### OrthoController (687 lines)

**Purpose:** Coordinate orthorectification UI

**Responsibilities:**
- GCP digitization workflow
- Orthorectification process coordination
- Preview updates
- Ortho table management

**Model:** OrthoModel (GCPs, homography, RMSE, is_rectified)

**Service:** OrthorectificationService

---

#### GridController (595 lines)

**Purpose:** Coordinate grid generation UI

**Responsibilities:**
- Grid preparation workflow
- Grid generation coordination
- Grid table management
- Grid visualization

**Model:** GridModel (grid points, parameters, is_generated)

**Service:** GridService

---

## Model Layer

### Model Design Principles

1. **State Container**
   - Hold application state
   - Provide properties for controlled access
   - Validate state on setters

2. **Qt Signals**
   - Inherit from `QObject`
   - Emit signals when state changes
   - Signal naming: `{property}_changed`

3. **No Business Logic**
   - Only validation, no calculations
   - Simple transformations OK (e.g., formatting)

4. **Serializable**
   - State should be serializable to JSON
   - Support project save/load

### Model Examples

#### VideoModel (216 lines)

**State:**
- `video_path` - Current video file path
- `position` - Current playback position (ms)
- `duration` - Video duration (ms)
- `is_playing` - Playback state
- `clip_start_time` - Clip start (s)
- `clip_end_time` - Clip end (s)

**Signals:**
- `video_loaded` - New video loaded
- `position_changed` - Position updated
- `duration_changed` - Duration determined
- `playing_state_changed` - Play/pause toggled

---

#### OrthoModel (316 lines)

**State:**
- `gcp_points` - Ground control points
- `homography_matrix` - Transformation matrix
- `rmse` - Reprojection error
- `is_rectified` - Rectification status
- `calibration_image_path` - Calibration image

**Signals:**
- `gcp_added` - New GCP digitized
- `gcp_removed` - GCP deleted
- `homography_computed` - Transform calculated
- `rectification_complete` - Rectification done

---

## Testing Strategy

### Test Pyramid

```
        ┌───────────┐
        │ GUI Tests │ (Few - Manual QA)
        └───────────┘
       ┌─────────────────┐
       │Integration Tests│ (Some - Workflows)
       └─────────────────┘
    ┌─────────────────────────┐
    │     Service Tests       │ (Many - TDD)
    └─────────────────────────┘
 ┌────────────────────────────────┐
 │    Core Module Tests           │ (Many - Unit tests)
 └────────────────────────────────┘
```

### Service Testing (TDD Approach)

**Example:** DischargeService

```python
# tests/test_services/test_discharge_service.py

@pytest.fixture
def discharge_service():
    """Create service instance."""
    return DischargeService()

@pytest.fixture
def mock_xs_survey():
    """Mock cross-section survey."""
    mock = Mock()
    mock.get_pixel_xs = Mock(return_value=(
        np.array([0.0, 5.0, 10.0]),  # stations
        np.array([10.0, 8.0, 7.5])    # elevations
    ))
    return mock

class TestGetStationAndDepth:
    def test_basic_extraction(self, discharge_service, mock_xs_survey):
        """Test station and depth extraction."""
        grid_points = np.array([[0, 0], [10, 0]])
        wse = 9.0

        stations, depths = discharge_service.get_station_and_depth(
            mock_xs_survey, grid_points, wse
        )

        assert len(stations) == 3
        assert np.all(depths == wse - np.array([10.0, 8.0, 7.5]))
```

**Benefits:**
- Tests written before implementation
- Clear specification of expected behavior
- Confidence in refactoring
- Living documentation

### Integration Testing

**Example:** Complete discharge workflow

```python
# tests/test_integration/test_discharge_workflow.py

def test_complete_discharge_calculation():
    """Test full discharge calculation workflow."""
    # Load test project
    project = ProjectService().load_project('tests/data/test_project.ivy')

    # Create services
    discharge_service = DischargeService()

    # Process
    df = discharge_service.create_discharge_dataframe(
        stations, depths, velocities, alpha=0.85
    )
    result = discharge_service.compute_discharge(df)
    uncertainty = discharge_service.compute_uncertainty(
        result['discharge_results'],
        result['total_discharge'],
        rmse, width
    )

    # Verify
    assert result['total_discharge'] > 0
    assert uncertainty['u_iso'] > 0
```

---

## Adding New Features

### Workflow for New Feature

1. **Identify Layer**
   - Is it a new calculation? → Service
   - Is it UI coordination? → Controller
   - Is it state management? → Model

2. **Create Service (if needed)**
   ```bash
   # 1. Create test file
   touch tests/test_services/test_new_service.py

   # 2. Write tests (TDD)
   # ... define expected behavior ...

   # 3. Create service file
   touch services/new_service.py

   # 4. Implement to pass tests
   # ... implement methods ...

   # 5. Integrate into controller
   # ... update controller to use service ...
   ```

3. **Create Controller (if needed)**
   ```bash
   # 1. Create controller file
   touch gui/controllers/new_controller.py

   # 2. Create model file
   touch gui/models/new_model.py

   # 3. Implement controller
   # ... orchestration logic ...

   # 4. Integrate into ivy.py
   # ... initialize controller in __init__ ...
   ```

4. **Update View**
   - Connect UI widgets to controller methods
   - Subscribe to model signals
   - Update widgets on signal emission

5. **Test**
   - Run service tests: `pytest tests/test_services/test_new_service.py`
   - Run all tests: `pytest`
   - Manual QA of UI workflow

---

## Code Examples

### Example 1: Adding a New Calculation

**Goal:** Add water quality index calculation

**Step 1: Create Service with Tests (TDD)**

```python
# tests/test_services/test_water_quality_service.py

class TestWaterQualityService:
    def test_compute_wqi_basic(self):
        """Test basic WQI calculation."""
        service = WaterQualityService()
        params = {
            'do': 8.5,      # mg/L
            'ph': 7.2,
            'turbidity': 5.0  # NTU
        }

        wqi = service.compute_wqi(params)

        assert 0 <= wqi <= 100
        assert wqi > 70  # Good quality

# services/water_quality_service.py

class WaterQualityService:
    def compute_wqi(self, params: dict) -> float:
        """Compute Water Quality Index."""
        # Implementation
        return wqi
```

**Step 2: Integrate into Controller**

```python
# controllers/water_quality_controller.py

class WaterQualityController(BaseController):
    def __init__(self, main_window, wq_model, wq_service):
        self.main_window = main_window
        self.wq_model = wq_model
        self.wq_service = wq_service
        self._connect_signals()

    def on_compute_wqi_clicked(self):
        """Compute WQI when user clicks button."""
        # Get data from UI
        params = self._get_params_from_ui()

        # Compute via service
        wqi = self.wq_service.compute_wqi(params)

        # Update model
        self.wq_model.wqi_value = wqi
```

**Step 3: Update View**

```python
# ivy.py

def __init__(self):
    # ... existing code ...

    # Create WQ components
    self.wq_service = WaterQualityService()
    self.wq_model = WaterQualityModel()
    self.wq_controller = WaterQualityController(
        self, self.wq_model, self.wq_service
    )

    # Connect UI
    self.pushButton_compute_wqi.clicked.connect(
        self.wq_controller.on_compute_wqi_clicked
    )
    self.wq_model.wqi_value_changed.connect(
        self.on_wqi_updated
    )

def on_wqi_updated(self, wqi: float):
    """Update UI when WQI is computed."""
    self.label_wqi.setText(f"WQI: {wqi:.1f}")
```

---

### Example 2: Adding State to Existing Feature

**Goal:** Track video playback history

**Step 1: Update Model**

```python
# models/video_model.py

class VideoModel(QObject):
    playback_history_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._playback_history = []

    def add_to_history(self, video_path: Path):
        """Add video to playback history."""
        if video_path not in self._playback_history:
            self._playback_history.insert(0, video_path)
            self._playback_history = self._playback_history[:10]  # Keep last 10
            self.playback_history_changed.emit(self._playback_history)
```

**Step 2: Update Controller**

```python
# controllers/video_controller.py

def load_video(self, path: Path):
    """Load video and add to history."""
    # ... existing loading code ...

    # Add to history
    self.video_model.add_to_history(path)
```

**Step 3: Update View**

```python
# ivy.py

def __init__(self):
    # ... existing code ...

    self.video_model.playback_history_changed.connect(
        self.update_recent_videos_menu
    )

def update_recent_videos_menu(self, history: list):
    """Update recent videos menu."""
    self.menu_recent_videos.clear()
    for video_path in history:
        action = self.menu_recent_videos.addAction(video_path.name)
        action.triggered.connect(
            lambda checked, p=video_path: self.video_controller.load_video(p)
        )
```

---

## Best Practices

### Services

✅ **DO:**
- Keep services stateless or minimal state
- Use type hints for all methods
- Write comprehensive docstrings
- Write tests first (TDD)
- Return Python native types
- Handle errors gracefully

❌ **DON'T:**
- Import PyQt5 in services
- Store mutable state
- Access UI widgets
- Print to console (use logging)
- Return Qt types

---

### Controllers

✅ **DO:**
- Orchestrate, don't calculate
- Update Models, not widgets
- Handle all Qt signals/slots
- Use descriptive slot names
- Document signal connections

❌ **DON'T:**
- Perform calculations
- Directly manipulate widget values
- Store state (use Models)
- Call other controllers directly
- Mix feature areas

---

### Models

✅ **DO:**
- Emit signals on state changes
- Validate state in setters
- Use properties for access
- Keep state serializable
- Document signals

❌ **DON'T:**
- Perform calculations
- Access widgets
- Call services directly
- Hold references to controllers
- Mutate state without signals

---

### Views

✅ **DO:**
- Forward events to controllers
- React to model signals
- Keep UI logic minimal
- Use Qt Designer for layouts

❌ **DON'T:**
- Perform calculations
- Manage state directly
- Call services directly
- Implement business logic

---

### Testing

✅ **DO:**
- Write tests first (TDD)
- Test edge cases
- Use fixtures for reusable data
- Mock dependencies
- Test one thing per test
- Use descriptive test names

❌ **DON'T:**
- Test implementation details
- Write tests after the fact
- Skip edge cases
- Test multiple things in one test
- Use real file I/O (use mocks)

---

## Migration Guide

### For Existing Code

When working with legacy code in ivy.py:

1. **Identify Business Logic**
   - Look for calculations, algorithms, data processing

2. **Extract to Service**
   - Create service class
   - Write tests first
   - Move logic to service
   - Update ivy.py to call service

3. **Extract UI Coordination**
   - Identify groups of related methods
   - Create controller
   - Create model for state
   - Move methods to controller
   - Update ivy.py to delegate

4. **Test Thoroughly**
   - Run unit tests
   - Run integration tests
   - Manual QA

### For New Code

Always follow the architecture:

1. **New Calculation?** → Create Service + Tests
2. **New UI Feature?** → Create Controller + Model
3. **New State?** → Update Model
4. **New UI Widget?** → Update View, connect to Controller

---

## Appendix

### Key Files

- `ivy.py` - Main window (5,817 lines)
- `services/__init__.py` - Service exports
- `controllers/__init__.py` - Controller exports
- `models/__init__.py` - Model exports

### Naming Conventions

- **Services:** `{Feature}Service` (e.g., `VideoService`)
- **Controllers:** `{Feature}Controller` (e.g., `VideoController`)
- **Models:** `{Feature}Model` (e.g., `VideoModel`)
- **Signals:** `{property}_changed` (e.g., `position_changed`)
- **Slots:** `on_{action}_{event}` (e.g., `on_play_clicked`)

### Dependencies

**Service Dependencies:**
- Python 3.11+
- NumPy, SciPy, pandas
- OpenCV, PIL, scikit-image
- Core modules (stiv, orthorectification, etc.)

**Controller/Model Dependencies:**
- PyQt5
- Services
- Models

### Further Reading

- [Qt Model/View Programming](https://doc.qt.io/qt-5/model-view-programming.html)
- [PyQt5 Signals and Slots](https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html)
- [MVP Pattern](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93presenter)
- [Service Layer Pattern](https://martinfowler.com/eaaCatalog/serviceLayer.html)

---

**Document Version:** 2.0
**Last Updated:** 2025-11-19
**Maintained By:** Development Team
**Status:** ✅ Production - Post Refactoring
