# IVyTools Codebase Refactoring Plan

**Date:** 2025-11-17
**Prepared by:** Claude (Automated Analysis)
**Scope:** image_velocimetry_tools package refactoring

---

## Executive Summary

The IVyTools codebase has grown to a state where the main GUI module (`gui/ivy.py`) has become a **7,033-line monolithic "God Object"** with 161 methods. While the core business logic is well-separated into focused modules, the GUI layer suffers from poor separation of concerns, mixing UI event handling with business logic, data processing, and state management.

This refactoring plan provides a phased approach to improve code maintainability, testability, and developer experience **without breaking existing functionality or changing the UI/UX**.

**Key Metrics:**
- **Main GUI file:** 7,033 lines (single class with 161 methods)
- **Total GUI code:** ~13,000 lines across 20+ files
- **Core business logic:** ~10,000 lines across 15+ modules (well-structured)
- **Test coverage:** 113 tests across 7 files (business logic only, no GUI tests)

**Critical Finding:** The `IvyTools` class in `gui/ivy.py` is a textbook "God Object" anti-pattern that should be the primary focus of refactoring efforts.

---

## Current State Analysis

### Strengths

1. **Well-Separated Core Business Logic**
   - `stiv.py` (1,175 lines) - Pure STIV algorithm implementation
   - `orthorectification.py` (1,541 lines) - Camera transformation logic
   - `common_functions.py` (872 lines) - Reusable utilities
   - `image_processing_tools.py` (1,087 lines) - Image processing without UI
   - These modules are testable, focused, and have minimal dependencies

2. **Tab Classes Already Modular**
   - `DischargeTab`, `STIVTab`, `CrossSectionGeometry` classes follow good composition patterns
   - They accept `ivy_framework` as a dependency, enabling loose coupling
   - Could be further improved but provide a solid foundation

3. **Good Test Coverage for Business Logic**
   - 113 test methods covering core algorithms
   - Tests are well-organized by module
   - Use real test data for validation

4. **Clear Package Structure**
   - Separation between `gui/` and core modules
   - Use of Qt Designer `.ui` files for dialog definitions
   - Logical naming conventions

### Weaknesses

1. **Monolithic Main Window Class (`gui/ivy.py` - 7,033 lines)**

   **Problems:**
   - **God Object Anti-pattern:** Single class responsible for everything
   - **161 methods** in one class (should be ~20-30 max)
   - **Mixed concerns:** UI, business logic, state, file I/O, calculations
   - **Impossible to unit test** individual features without full GUI setup
   - **Cognitive overload:** Developers must understand 7,000+ lines to modify anything
   - **Merge conflicts:** Any two developers working on GUI will conflict
   - **No clear ownership:** Every part of the app is in one file

   **Examples of Mixed Concerns:**
   ```python
   # Lines 3500-3570: Homography calculations in UI class
   # Lines 4119-4288: Grid generation logic in UI class
   # Lines 2399-2542: Video playback AND FFmpeg command building
   # Lines 4643-4768: CSV loading AND table widget manipulation
   ```

2. **Business Logic Embedded in GUI Code**

   **gui/ivy.py contains:**
   - Homography matrix calculations
   - Coordinate transformations
   - Video frame extraction logic
   - Image stack processing
   - Grid generation algorithms
   - CSV file parsing and validation

   **Should be in:** Separate service/controller classes or existing business logic modules

3. **State Management Issues**
   - Global state scattered across 161 methods
   - No clear data model or state container
   - Difficult to track what state exists and when it changes
   - Side effects everywhere (methods modify multiple attributes)

4. **Testability Problems**
   - **Zero GUI tests** (understandable but problematic)
   - Business logic in GUI = untestable business logic
   - No integration tests between UI and core modules
   - Cannot test user workflows without manual QA

5. **Code Navigation Challenges**
   - Finding specific functionality requires searching 7,000 lines
   - Related methods scattered throughout the file
   - No logical grouping or organization within the class
   - Method names don't follow consistent patterns

### Architecture Comparison

**Current Architecture:**
```
┌─────────────────────────────────────────────┐
│         IvyTools (7,033 lines)              │
│  ┌─────────────────────────────────────┐   │
│  │  UI Event Handlers                  │   │
│  │  Business Logic                     │   │
│  │  State Management                   │   │
│  │  File I/O                           │   │
│  │  Calculations                       │   │
│  │  Video Processing                   │   │
│  │  Grid Generation                    │   │
│  │  Project Management                 │   │
│  │  Settings                           │   │
│  │  ... 153 more methods ...          │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
         ↓ calls directly
┌─────────────────────────────────────────────┐
│       Core Business Logic Modules           │
│  (stiv, orthorectification, etc.)           │
└─────────────────────────────────────────────┘
```

**Desired Architecture (MVP Pattern):**
```
┌─────────────────────────────────────────────┐
│    IvyToolsMainWindow (~500 lines)          │
│    - Window setup & orchestration           │
│    - Tab management                         │
│    - Menu/toolbar coordination              │
└─────────────────────────────────────────────┘
         ↓ delegates to
┌─────────────────────────────────────────────┐
│         Controllers/Presenters              │
│  ┌──────────────┐  ┌──────────────┐        │
│  │VideoCtrl     │  │OrthoCtrl     │  ...   │
│  │ProjectCtrl   │  │GridCtrl      │        │
│  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────┘
         ↓ uses
┌─────────────────────────────────────────────┐
│          Services (Business Logic)          │
│  ┌──────────────┐  ┌──────────────┐        │
│  │VideoSvc      │  │OrthoSvc      │  ...   │
│  │ProjectSvc    │  │GridSvc       │        │
│  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────┘
         ↓ uses
┌─────────────────────────────────────────────┐
│       Core Business Logic Modules           │
│  (stiv, orthorectification, etc.)           │
└─────────────────────────────────────────────┘
```

---

## Refactoring Recommendations

### Priority 1: Critical - Decompose `gui/ivy.py`

**Objective:** Break the 7,033-line God Object into focused, testable components.

#### Recommendation 1.1: Extract Controllers (Presenters)

Create controller classes for each major feature area:

```
gui/
├── ivy.py (reduced to ~500 lines - main window orchestration only)
├── controllers/
│   ├── __init__.py
│   ├── video_controller.py         # Video playback, FFmpeg operations
│   ├── project_controller.py       # Project save/load/management
│   ├── orthorectification_controller.py  # Ortho UI coordination
│   ├── grid_controller.py          # Grid generation UI coordination
│   ├── settings_controller.py      # Settings management
│   └── image_stack_controller.py   # Image stack processing coordination
```

**Responsibility:** Each controller handles:
- UI event routing for its feature area
- Coordination between UI widgets and services
- UI state updates based on service responses
- **NO business logic** - only orchestration

**Benefits:**
- Each controller is ~200-400 lines (manageable)
- Clear ownership of features
- Testable with mock services
- Parallel development possible

#### Recommendation 1.2: Extract Services (Business Logic)

Create service classes that contain business logic currently in `ivy.py`:

```
services/
├── __init__.py
├── video_service.py           # FFmpeg command building, video metadata
├── project_service.py         # Project serialization/deserialization
├── orthorectification_service.py  # Homography calculations, transformations
├── grid_service.py            # Grid generation algorithms
├── image_stack_service.py     # Image stack creation and processing
└── validation_service.py      # Data validation logic
```

**Responsibility:** Each service handles:
- Pure business logic (no Qt dependencies)
- Data transformations
- Calculations and algorithms
- Validation rules
- File I/O (delegated to existing file_management module)

**Benefits:**
- 100% unit testable (no GUI needed)
- Reusable from CLI or other interfaces
- Clear separation of concerns
- Can use existing core modules as dependencies

#### Recommendation 1.3: Extract State Management

Create model classes to hold application state:

```
models/
├── __init__.py
├── project_model.py           # Project state (paths, metadata, settings)
├── video_model.py             # Video state (current position, clip times, etc.)
├── ortho_model.py             # Orthorectification state (GCP points, homography)
├── grid_model.py              # Grid state (grid points, parameters)
└── app_state.py               # Global app state manager
```

**Responsibility:**
- Hold application state
- Emit signals when state changes (Qt signals for GUI updates)
- Validate state transitions
- Serialize/deserialize to JSON

**Benefits:**
- Single source of truth for state
- Clear state ownership
- Easier debugging (state in one place)
- Enables undo/redo in future

#### Recommendation 1.4: Refactored `ivy.py` Structure

The new `ivy.py` should be ~500 lines and look like:

```python
class IvyToolsMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main window - orchestration only"""

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialize state models
        self.app_state = AppState()
        self.project_model = ProjectModel()
        self.video_model = VideoModel()

        # Initialize services
        self.video_service = VideoService()
        self.project_service = ProjectService()
        self.ortho_service = OrthorectificationService()

        # Initialize controllers
        self.video_controller = VideoController(self, self.video_model, self.video_service)
        self.project_controller = ProjectController(self, self.project_model, self.project_service)
        self.ortho_controller = OrthoController(self, self.ortho_model, self.ortho_service)

        # Initialize tab components (already modular!)
        self.discharge_tab = DischargeTab(self)
        self.stiv_tab = STIVTab(self)
        self.xs_geometry = CrossSectionGeometry(self)

        # Setup UI connections
        self._connect_signals()
        self._setup_tabs()
        self._setup_toolbar()

    def _connect_signals(self):
        """Connect UI signals to controllers"""
        # Delegate to controllers
        pass

    # Only high-level orchestration methods here
    # All feature logic delegated to controllers
```

### Priority 2: High - Extract Business Logic from Tab Classes

**Objective:** Remove business logic from `DischargeTab`, `STIVTab`, `CrossSectionGeometry`.

#### Recommendation 2.1: Extract from DischargeTab

**Current issues in `gui/discharge.py`:**
- Lines 200-350: Discharge calculations mixed with table updates
- Lines 400-500: Uncertainty analysis mixed with UI updates

**Solution:**
```
services/
└── discharge_service.py  # Extract discharge calculations, uncertainty analysis

gui/discharge.py (reduced to ~400 lines)
- Only table widget management
- Delegates calculations to discharge_service
```

#### Recommendation 2.2: Extract from STIVTab

**Current issues in `gui/stiv_processor.py`:**
- Lines 86-150: Calls STIV algorithms directly
- Lines 200-300: Result processing mixed with UI

**Solution:**
```
services/
└── stiv_coordination_service.py  # STIV parameter management, result processing

gui/stiv_processor.py (reduced to ~500 lines)
- Only UI widget management
- Delegates to stiv_coordination_service
- stiv_coordination_service uses existing stiv.py module
```

#### Recommendation 2.3: Extract from CrossSectionGeometry

**Current issues in `gui/xsgeometry.py`:**
- Heavy integration with AreaComp library
- Cross-section calculations mixed with plotting

**Solution:**
```
services/
└── cross_section_service.py  # Cross-section calculations, AreaComp integration

gui/xsgeometry.py (reduced to ~700 lines)
- Only plot and table management
- Delegates to cross_section_service
```

### Priority 3: Medium - Improve Testing

**Objective:** Increase test coverage and add integration tests.

#### Recommendation 3.1: Add Service Tests

Create tests for new service classes:

```
tests/
├── test_services/
│   ├── test_video_service.py
│   ├── test_project_service.py
│   ├── test_ortho_service.py
│   ├── test_grid_service.py
│   └── test_discharge_service.py
```

**Target:** 80%+ coverage of service layer

#### Recommendation 3.2: Add Integration Tests

Create integration tests that verify workflows:

```
tests/
├── test_integration/
│   ├── test_project_workflow.py       # Create, save, load project
│   ├── test_video_workflow.py         # Load video, extract frames
│   ├── test_ortho_workflow.py         # GCP digitization, rectification
│   └── test_discharge_workflow.py     # Complete discharge computation
```

**Target:** Cover major user workflows

#### Recommendation 3.3: Add Controller Tests (Optional)

If resources allow, add tests for controllers:

```
tests/
├── test_controllers/
│   ├── test_video_controller.py
│   ├── test_project_controller.py
│   └── test_ortho_controller.py
```

Use mock services to test controller logic without full GUI.

### Priority 4: Low - Code Organization Improvements

#### Recommendation 4.1: Consistent Naming Conventions

**Current inconsistencies:**
- Some methods use `snake_case`, others `camelCase` (Qt influence)
- Inconsistent signal naming (`signal_stderr` vs `pointsExistSignal`)

**Solution:**
- Standardize on `snake_case` for Python code (PEP 8)
- Keep Qt slot names as-is for compatibility
- Document naming convention in CONTRIBUTING.md

#### Recommendation 4.2: Type Hints

Add type hints to improve IDE support and catch errors:

```python
def process_stiv_exhaustive(
    self,
    progress_callback: Callable[[int], None]
) -> Tuple[np.ndarray, np.ndarray]:
    """Process STIV with type hints"""
    pass
```

**Target:** Add type hints to all new code, gradually to existing code

#### Recommendation 4.3: Docstring Standardization

**Current:** Mix of Google, NumPy, and minimal docstrings

**Solution:** Standardize on Google style (already used in some modules):

```python
def calculate_discharge(
    self,
    velocity: float,
    area: float
) -> float:
    """Calculate discharge from velocity and area.

    Args:
        velocity: Mean velocity in m/s
        area: Cross-sectional area in m²

    Returns:
        Discharge in m³/s

    Raises:
        ValueError: If velocity or area is negative
    """
    pass
```

---

## Detailed Implementation Plan

### Phase 1: Foundation (Week 1-2)

**Goal:** Set up infrastructure for refactoring without breaking anything.

#### Tasks:

1. **Create package structure**
   ```bash
   mkdir -p image_velocimetry_tools/gui/controllers
   mkdir -p image_velocimetry_tools/gui/models
   mkdir -p image_velocimetry_tools/services
   touch image_velocimetry_tools/gui/controllers/__init__.py
   touch image_velocimetry_tools/gui/models/__init__.py
   touch image_velocimetry_tools/services/__init__.py
   ```

2. **Create base classes**
   - `BaseController` - Common controller functionality
   - `BaseService` - Common service functionality
   - `BaseModel` - Common model functionality with Qt signals

3. **Set up testing infrastructure**
   - Create `tests/test_services/` directory
   - Create `tests/test_controllers/` directory
   - Create `tests/test_integration/` directory
   - Set up pytest fixtures for common test data

4. **Run existing tests to establish baseline**
   ```bash
   pytest image_velocimetry_tools/tests/ -v
   ```
   **Success criteria:** All 113 tests pass

### Phase 2: Extract First Service (Week 2-3)

**Goal:** Prove the pattern works with a low-risk extraction.

**Target:** `VideoService` (FFmpeg operations, video metadata)

#### Tasks:

1. **Create `services/video_service.py`**
   - Extract FFmpeg command building from `ivy.py`
   - Extract video metadata parsing
   - Extract clip time validation
   - **NO Qt dependencies** (use Python types only)

2. **Write comprehensive tests**
   - `tests/test_services/test_video_service.py`
   - Test FFmpeg command generation
   - Test metadata parsing
   - Test validation logic
   - **Target:** 90%+ coverage

3. **Update `ivy.py` to use `VideoService`**
   - Create `self.video_service = VideoService()` in `__init__`
   - Replace inline logic with service calls
   - Keep UI update logic in `ivy.py`

4. **Run all tests**
   ```bash
   pytest image_velocimetry_tools/tests/ -v
   ```
   **Success criteria:** All tests pass (113 + new service tests)

5. **Manual QA**
   - Load video
   - Extract frames
   - Create clip
   - Verify no behavior changes

### Phase 3: Extract First Controller (Week 3-4)

**Goal:** Prove the controller pattern works.

**Target:** `VideoController` (video playback UI coordination)

#### Tasks:

1. **Create `controllers/video_controller.py`**
   - Extract video playback methods from `ivy.py` (~30 methods)
   - Extract FFmpeg thread management
   - Extract video position/duration handling
   - Keep reference to main window for widget access

2. **Create `models/video_model.py`**
   - Create state model for video (path, position, duration, clip times)
   - Emit Qt signals on state changes
   - Validation logic for state transitions

3. **Update `ivy.py` to use `VideoController`**
   - Create controller in `__init__`
   - Connect UI signals to controller methods
   - Subscribe to model signals for UI updates

4. **Write tests**
   - `tests/test_controllers/test_video_controller.py`
   - Mock the service and model
   - Test controller orchestration logic

5. **Run all tests + Manual QA**

**Success criteria:**
- All tests pass
- Video functionality unchanged
- `ivy.py` reduced by ~500 lines

### Phase 4: Extract Remaining Services (Week 4-6)

**Goal:** Extract all business logic from `ivy.py`.

**Targets:**
1. `ProjectService` (Week 4)
   - Project save/load
   - Project validation
   - JSON serialization

2. `OrthorectificationService` (Week 5)
   - Homography calculations
   - Coordinate transformations
   - GCP validation
   - RMSE estimation

3. `GridService` (Week 5-6)
   - Grid generation algorithms
   - Grid validation
   - Grid serialization

4. `ImageStackService` (Week 6)
   - Image stack creation
   - Image preprocessing coordination

#### Process for each service:

1. Create service file
2. Write tests (TDD approach)
3. Extract logic from `ivy.py`
4. Update `ivy.py` to use service
5. Run all tests + manual QA
6. **Commit after each service** (incremental progress)

**Success criteria per service:**
- All tests pass
- No functionality changes
- Service has 80%+ test coverage

### Phase 5: Extract Remaining Controllers (Week 6-8)

**Goal:** Complete the controller extraction.

**Targets:**
1. `ProjectController` (Week 6-7)
   - New project, open project, save project
   - Project structure management
   - Recent projects

2. `OrthoController` (Week 7)
   - GCP digitization UI
   - Orthorectification UI
   - Ortho table management

3. `GridController` (Week 7-8)
   - Grid preparation UI
   - Grid generation UI
   - Grid table management

4. `SettingsController` (Week 8)
   - Settings dialog
   - Units conversion
   - Preferences

#### Process for each controller:

1. Create controller file
2. Create corresponding model file
3. Write tests (optional but recommended)
4. Extract methods from `ivy.py`
5. Update `ivy.py` to use controller
6. Run all tests + manual QA
7. **Commit after each controller**

**Success criteria:**
- `ivy.py` reduced to ~500-800 lines
- All tests pass
- No functionality changes

### Phase 6: Extract Business Logic from Tab Classes (Week 8-10)

**Goal:** Clean up tab classes.

**Targets:**

1. **DischargeTab** (Week 8-9)
   - Extract `DischargeService`
   - Update `DischargeTab` to use service
   - Add tests for `DischargeService`

2. **STIVTab** (Week 9)
   - Extract `StivCoordinationService`
   - Update `STIVTab` to use service
   - Add tests for service

3. **CrossSectionGeometry** (Week 9-10)
   - Extract `CrossSectionService`
   - Update `CrossSectionGeometry` to use service
   - Add tests for service

**Success criteria:**
- Each tab class reduced by 30-50%
- All business logic testable
- All tests pass

### Phase 7: Testing & Documentation (Week 10-11)

**Goal:** Ensure quality and knowledge transfer.

#### Tasks:

1. **Integration tests**
   - Write 5-10 integration tests covering major workflows
   - Test with real data from `tests/img_seq/`

2. **Increase unit test coverage**
   - Target 80%+ coverage for services
   - Target 60%+ coverage for controllers

3. **Documentation**
   - Update README with new architecture
   - Create ARCHITECTURE.md explaining the pattern
   - Create CONTRIBUTING.md with coding standards
   - Add docstrings to all new classes and methods

4. **Code review**
   - Review all changes for consistency
   - Ensure naming conventions followed
   - Verify no dead code remains

5. **Performance testing**
   - Ensure no performance regressions
   - Profile common operations

**Success criteria:**
- All tests pass (target: 200+ tests)
- Test coverage >75% overall
- Documentation complete
- No performance regressions

### Phase 8: Cleanup & Polish (Week 11-12)

**Goal:** Finalize the refactoring.

#### Tasks:

1. **Remove dead code**
   - Identify and remove unused methods
   - Remove commented-out code
   - Clean up imports

2. **Consistency pass**
   - Ensure naming conventions consistent
   - Ensure docstring style consistent
   - Ensure file organization consistent

3. **Type hints**
   - Add type hints to all new code
   - Add type hints to frequently-used existing code

4. **Final QA**
   - Complete manual testing of all features
   - Test on Windows, macOS, Linux (if possible)
   - Test with real user data

5. **Prepare for merge**
   - Squash commits if desired
   - Write comprehensive commit message
   - Update CHANGELOG.md

**Success criteria:**
- All tests pass
- All features work as before
- Code is clean and consistent
- Documentation is complete

---

## Risk Mitigation Strategy

### Risk 1: Breaking Existing Functionality

**Likelihood:** Medium
**Impact:** High

**Mitigation:**
- ✅ Incremental refactoring (one service/controller at a time)
- ✅ Run all tests after each change
- ✅ Manual QA after each change
- ✅ Commit after each successful extraction
- ✅ Keep feature branches small and focused
- ✅ Use feature flags if needed to enable/disable refactored code

### Risk 2: Test Suite Insufficient to Catch Regressions

**Likelihood:** Medium
**Impact:** High

**Mitigation:**
- ✅ Write new tests BEFORE extracting code (TDD)
- ✅ Add integration tests for major workflows
- ✅ Extensive manual QA checklist
- ✅ Beta testing with select users before release
- ✅ Keep old code commented out temporarily for comparison

### Risk 3: Qt Signal/Slot Connections Break

**Likelihood:** Medium
**Impact:** Medium

**Mitigation:**
- ✅ Carefully track all signal/slot connections
- ✅ Document connections in refactored classes
- ✅ Test UI interactions thoroughly
- ✅ Use Qt's connection debugging features
- ✅ Keep connection setup in main window initially

### Risk 4: Performance Regression

**Likelihood:** Low
**Impact:** Medium

**Mitigation:**
- ✅ Profile before and after refactoring
- ✅ Avoid unnecessary object creation
- ✅ Reuse existing service instances
- ✅ Monitor memory usage
- ✅ Test with large datasets

### Risk 5: Incomplete Refactoring (Time Runs Out)

**Likelihood:** Medium
**Impact:** Low

**Mitigation:**
- ✅ Prioritize high-impact extractions first
- ✅ Each phase produces working code (can stop anytime)
- ✅ Focus on `ivy.py` decomposition as primary goal
- ✅ Tab class refactoring is optional (nice-to-have)
- ✅ Documentation and polish can be deferred

### Risk 6: Merge Conflicts During Long Refactoring

**Likelihood:** High (if active development)
**Impact:** Medium

**Mitigation:**
- ✅ Communicate refactoring plan with team
- ✅ Coordinate on feature freeze or separate branch
- ✅ Merge main into refactoring branch frequently
- ✅ Keep refactoring branch short-lived (merge within 2-3 weeks if possible)
- ✅ Break into smaller PRs if possible

### Risk 7: Developer Resistance to New Pattern

**Likelihood:** Low
**Impact:** Medium

**Mitigation:**
- ✅ Document the pattern clearly (ARCHITECTURE.md)
- ✅ Provide examples of each pattern
- ✅ Pair programming for first few implementations
- ✅ Code review to ensure consistency
- ✅ Show benefits: easier testing, clearer code, parallel development

---

## Success Metrics

### Quantitative Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Largest file size | 7,033 lines | <1,000 lines | Line count |
| Methods per class (avg) | 161 (IvyTools) | <30 | Method count |
| Test count | 113 | 200+ | pytest count |
| Test coverage | ~50% (business logic only) | >75% overall | pytest-cov |
| Cyclomatic complexity (ivy.py) | Very High | Medium | radon cc |
| GUI classes with business logic | 4 | 0 | Manual inspection |
| Untestable business logic (LOC) | ~3,000 | <500 | Estimate |

### Qualitative Metrics

- ✅ Developer can find relevant code in <2 minutes
- ✅ New feature can be added without modifying `ivy.py`
- ✅ Business logic can be tested without GUI
- ✅ Two developers can work on different features without conflicts
- ✅ Onboarding a new developer takes <2 days to understand architecture

---

## Implementation Notes

### Recommended Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b refactor/decompose-ivy-gui
   ```

2. **For each service/controller extraction:**
   ```bash
   # 1. Create files
   touch image_velocimetry_tools/services/video_service.py
   touch image_velocimetry_tools/tests/test_services/test_video_service.py

   # 2. Write tests (TDD)
   # ... write tests ...
   pytest image_velocimetry_tools/tests/test_services/test_video_service.py

   # 3. Implement service
   # ... implement ...
   pytest image_velocimetry_tools/tests/test_services/test_video_service.py

   # 4. Integrate into ivy.py
   # ... update ivy.py ...

   # 5. Run all tests
   pytest image_velocimetry_tools/tests/ -v

   # 6. Manual QA
   # ... test video features ...

   # 7. Commit
   git add .
   git commit -m "refactor: extract VideoService from ivy.py

   - Create VideoService with FFmpeg command generation
   - Create tests for VideoService (15 tests, 90% coverage)
   - Update ivy.py to use VideoService
   - Reduce ivy.py by 200 lines
   - All existing tests pass
   - Manual QA: video loading, clip creation, frame extraction"
   ```

3. **After each phase:**
   ```bash
   # Run full test suite
   pytest image_velocimetry_tools/tests/ -v --cov=image_velocimetry_tools

   # Manual QA checklist
   # - Load project
   # - Load video
   # - Extract frames
   # - Digitize GCPs
   # - Orthorectify
   # - Run STIV
   # - Compute discharge
   # - Generate report
   # - Save project

   # Commit phase completion
   git commit -m "refactor: complete Phase X - <description>"
   ```

### Code Style Guidelines

**Service Classes:**
```python
# services/video_service.py
"""Business logic for video operations."""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

class VideoService:
    """Service for video processing operations.

    This service handles FFmpeg command generation, video metadata
    parsing, and video clip validation. It has no Qt dependencies
    and can be used from CLI or other interfaces.
    """

    def __init__(self):
        """Initialize the video service."""
        self.logger = logging.getLogger(__name__)

    def build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        **kwargs
    ) -> list[str]:
        """Build FFmpeg command for video clip extraction.

        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            start_time: Clip start time in seconds
            end_time: Clip end time in seconds
            **kwargs: Additional FFmpeg options

        Returns:
            List of command arguments for subprocess

        Raises:
            ValueError: If times are invalid
        """
        self._validate_clip_times(start_time, end_time)
        # Implementation...

    def _validate_clip_times(
        self,
        start_time: float,
        end_time: float
    ) -> None:
        """Validate clip start and end times."""
        if start_time < 0:
            raise ValueError("Start time cannot be negative")
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time")
```

**Controller Classes:**
```python
# controllers/video_controller.py
"""Controller for video playback UI."""

import logging
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSlot

class VideoController(QObject):
    """Controller for video playback and processing UI.

    This controller coordinates between the video UI widgets,
    video model (state), and video service (business logic).
    """

    def __init__(
        self,
        main_window,
        video_model,
        video_service
    ):
        """Initialize the video controller.

        Args:
            main_window: Reference to main window for widget access
            video_model: Video state model
            video_service: Video business logic service
        """
        super().__init__()
        self.main_window = main_window
        self.video_model = video_model
        self.video_service = video_service
        self.logger = logging.getLogger(__name__)

        self._connect_signals()

    def _connect_signals(self):
        """Connect UI signals to controller methods."""
        self.main_window.pushButton_play_video.clicked.connect(
            self.on_play_clicked
        )
        self.video_model.position_changed.connect(
            self.on_model_position_changed
        )

    @pyqtSlot()
    def on_play_clicked(self):
        """Handle play button click."""
        # Get current state from model
        if self.video_model.is_playing:
            self.video_model.pause()
        else:
            self.video_model.play()

    @pyqtSlot(int)
    def on_model_position_changed(self, position: int):
        """Update UI when video position changes.

        Args:
            position: New position in milliseconds
        """
        # Update UI widgets
        self.main_window.slider_video_position.setValue(position)
        self.main_window.label_current_time.setText(
            self._format_time(position)
        )
```

**Model Classes:**
```python
# models/video_model.py
"""Model for video state."""

from typing import Optional
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

class VideoModel(QObject):
    """Model representing video state.

    Signals:
        video_loaded: Emitted when a new video is loaded
        position_changed: Emitted when playback position changes
        duration_changed: Emitted when video duration is determined
        playing_state_changed: Emitted when play/pause state changes
    """

    video_loaded = pyqtSignal(Path)
    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    playing_state_changed = pyqtSignal(bool)

    def __init__(self):
        """Initialize the video model."""
        super().__init__()
        self._video_path: Optional[Path] = None
        self._position: int = 0
        self._duration: int = 0
        self._is_playing: bool = False

    @property
    def video_path(self) -> Optional[Path]:
        """Get the current video path."""
        return self._video_path

    @video_path.setter
    def video_path(self, path: Optional[Path]):
        """Set the video path."""
        self._video_path = path
        if path:
            self.video_loaded.emit(path)

    @property
    def position(self) -> int:
        """Get current position in milliseconds."""
        return self._position

    @position.setter
    def position(self, value: int):
        """Set current position in milliseconds."""
        if value != self._position:
            self._position = value
            self.position_changed.emit(value)

    @property
    def is_playing(self) -> bool:
        """Check if video is currently playing."""
        return self._is_playing

    def play(self):
        """Start video playback."""
        if not self._is_playing:
            self._is_playing = True
            self.playing_state_changed.emit(True)

    def pause(self):
        """Pause video playback."""
        if self._is_playing:
            self._is_playing = False
            self.playing_state_changed.emit(False)
```

### Testing Guidelines

**Service Tests (TDD):**
```python
# tests/test_services/test_video_service.py
"""Tests for VideoService."""

import pytest
from pathlib import Path
from image_velocimetry_tools.services.video_service import VideoService

class TestVideoService:
    """Tests for VideoService."""

    @pytest.fixture
    def service(self):
        """Create a VideoService instance."""
        return VideoService()

    def test_build_ffmpeg_command_basic(self, service):
        """Test basic FFmpeg command generation."""
        cmd = service.build_ffmpeg_command(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
            start_time=10.0,
            end_time=20.0
        )

        assert "ffmpeg" in cmd[0]
        assert "-i" in cmd
        assert "input.mp4" in cmd
        assert "output.mp4" in cmd
        assert "-ss" in cmd
        assert "-t" in cmd

    def test_validate_clip_times_invalid_negative(self, service):
        """Test validation rejects negative start time."""
        with pytest.raises(ValueError, match="cannot be negative"):
            service._validate_clip_times(-1.0, 10.0)

    def test_validate_clip_times_invalid_order(self, service):
        """Test validation rejects end time before start time."""
        with pytest.raises(ValueError, match="must be greater"):
            service._validate_clip_times(20.0, 10.0)
```

**Integration Tests:**
```python
# tests/test_integration/test_video_workflow.py
"""Integration tests for video workflows."""

import pytest
from pathlib import Path
from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.models.video_model import VideoModel

class TestVideoWorkflow:
    """Test complete video workflows."""

    @pytest.fixture
    def test_video(self, tmp_path):
        """Create a test video file."""
        # Use existing test video or create one
        return Path("image_velocimetry_tools/tests/test_video.mp4")

    def test_load_video_and_extract_clip(self, test_video, tmp_path):
        """Test loading a video and extracting a clip."""
        # Arrange
        video_model = VideoModel()
        video_service = VideoService()
        output_path = tmp_path / "clip.mp4"

        # Act
        video_model.video_path = test_video
        metadata = video_service.get_metadata(test_video)
        cmd = video_service.build_ffmpeg_command(
            input_path=test_video,
            output_path=output_path,
            start_time=0.0,
            end_time=5.0
        )

        # Run FFmpeg (if testing actual execution)
        # subprocess.run(cmd, check=True)

        # Assert
        assert video_model.video_path == test_video
        assert metadata["duration"] > 0
        assert "ffmpeg" in cmd[0]
        # assert output_path.exists()  # If actually running FFmpeg
```

---

## Alternative Approaches Considered

### Alternative 1: Complete Rewrite

**Approach:** Rewrite the GUI from scratch using modern patterns.

**Pros:**
- Clean slate, perfect architecture
- Could use modern frameworks (Qt for Python best practices)
- No technical debt

**Cons:**
- ❌ Extremely high risk
- ❌ Would take 6-12 months
- ❌ High chance of introducing bugs
- ❌ Users would notice changes
- ❌ Not feasible for active project

**Decision:** Rejected - too risky

### Alternative 2: Leave As-Is, Only Add New Code Properly

**Approach:** Don't refactor existing code, only new code follows good patterns.

**Pros:**
- Zero risk to existing functionality
- Minimal effort

**Cons:**
- ❌ Technical debt continues to grow
- ❌ New code must integrate with messy existing code
- ❌ Problem gets worse over time
- ❌ Eventually becomes unmaintainable

**Decision:** Rejected - kicks the can down the road

### Alternative 3: Gradual Refactoring (Selected Approach)

**Approach:** Incrementally extract services and controllers from `ivy.py`.

**Pros:**
- ✅ Low risk (one piece at a time)
- ✅ Can stop at any time with improved code
- ✅ Each step is testable
- ✅ Provides immediate benefits
- ✅ Team learns patterns gradually

**Cons:**
- Takes time (8-12 weeks)
- Requires discipline to follow through
- Code exists in mixed state during transition

**Decision:** Selected - best balance of risk and reward

---

## Conclusion

The IVyTools codebase has strong foundations with well-separated business logic, but the GUI layer has grown into an unmaintainable monolith. The proposed refactoring plan uses proven patterns (MVP, Service Layer) to decompose the 7,033-line `ivy.py` into focused, testable components.

**Key Benefits:**
- 🎯 **Maintainability:** Developers can find and modify code quickly
- 🧪 **Testability:** Business logic is 100% testable without GUI
- 🚀 **Velocity:** New features can be added without touching core files
- 👥 **Collaboration:** Multiple developers can work in parallel
- 📚 **Onboarding:** New developers can understand the architecture

**Timeline:** 8-12 weeks for complete refactoring (can stop early with partial benefits)

**Risk Level:** Low (incremental approach with comprehensive testing)

**Recommendation:** Proceed with Phase 1-5 as minimum viable refactoring (focus on `ivy.py` decomposition). Phases 6-8 are optional enhancements.

---

## Next Steps

1. **Review this plan** with the development team
2. **Get approval** for the approach and timeline
3. **Set up development environment** for refactoring branch
4. **Begin Phase 1** (Foundation setup)
5. **Establish regular check-ins** to review progress and adjust plan

**Questions to address:**
- Is the 8-12 week timeline acceptable?
- Should we stop after Phase 5 or continue to Phases 6-8?
- Are there any features we should avoid touching?
- Are there specific areas of highest pain that should be prioritized?
- Do we want to do this in one long-running branch or multiple smaller PRs?

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Status:** Proposal - Awaiting Approval
