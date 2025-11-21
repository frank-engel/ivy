# Image Velocimetry Application

## Project Overview
Python application for computing river discharge from video analysis.
Target users: Field technicians measuring river flow. Non-programmers, but talented and teachable.

## Tech Stack
- **Language**: Python 3.x
- **UI Framework**: Qt (PyQt/PySide)
- **Architecture**: Model-View-Presenter (MVP)

## Core Components
- **Services**: Business logic and data processing
- **Controllers**: Orchestrate between views and models
- **Models**: Data structures and state management
- **Views**: Qt UI components

## Key Directories
- `/image_velocimetry_tools/gui/models/` - Data models and domain entities
- `/image_velocimetry_tools/gui/dialogs` - Qt UI components
- `/image_velocimetry_tools/services/` - Business logic and processing
- `/image_velocimetry_tools/gui/controllers/` - Application flow control
- `/image_velocimetry_tools/gui/tests/` - Unit and integration tests

## Project File Structure
Current project files are ZIP-based (*.ivy) with the following structure:
- `/1-images/` - frames extracted from videos (f*.jpg), rectified frames remapped to "top down" view (t*.jpg), and Space-Time Images (STIV) made by the stiv module
- `/2-orthorectification/` - copy of the calibration_image used in rectification. Several CSV files containing outputs from the IVyTools user client app. Don't trust these as truth of state. 
- `/4-velocities/` - contains a CSV with the most recent space-time image velocimetry (STIV) results
- `/5-discharge/` - contians the Areacomp cross-section_ac3.mat file, the source for cross-section geometry. Also contains a summary discharge CSV. Don't trust the CSV as current state. This _is_ where you need to grab the cross-section geometry
- `project_data.json` - definitive JSON file with UI state, lots of variables describing data needed for processing workflow. Key data objects include:
  -  `discharge_results` - current state of the discharge points table, with each node's data and status
  -  `cross_section*` - various date about the current cross-section geometry in relationship the rectified image or the AC3 file
  -  `display_inits` - expected display units. note that internal data are always SI, but input to the system may be in English units
  -  `ffmpage_parameters` - settings applied to video processing
  -  `extractions_*` - settings concerning frame extraction specifics
  -  `mask_polygons` - array of pixel coords for and polygon masks used when creating grid nodes
  -  `rectification_parameters` - contains important information about the rectification process. 

Note the current state of the `project_data.json` is messy and needs improvement. Take care that variables used here are correctly applied.

## Dependencies
See `requirements.txt`, note that areacomp3 is pulled from a gitlab server, not pip