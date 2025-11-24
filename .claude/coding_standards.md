# Coding Standards

## Python Style
- PEP 8 compliance
- Type hints on all function signatures
- Docstrings for public APIs (Numpy style)
- Maximum line length: 100 characters

## Naming Conventions
- **Classes**: PascalCase (e.g., `BatchProcessor`, `CrossSectionService`)
- **Functions/Methods**: snake_case (e.g., `load_project_from_json`, `create_image_stack`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `IVY_ENV`)
- **Private members**: Leading underscore (e.g., `_connect_signals`)

## File Organization
```python
# Standard import order:
# 1. Standard library
# 2. Third-party packages
# 3. Local application imports

from typing import List, Dict, Optional
import json
from pathlib import Path

import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal

from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.services.project_service import ProjectService
```

## Testing
- Unit tests for all services and models
- Integration tests for workflows
- Use pytest framework
- Aim for >80% coverage on business logic

## Documentation
- README in each major module directory
- Inline comments for complex algorithms only
- API documentation auto-generated from docstrings

## Error Handling Pattern
```python
class DomainException(Exception):
    """Base exception for domain errors"""
    pass

class InvalidBatchFileError(DomainException):
    """Raised when batch file is malformed"""
    pass

# In services:
def process_batch(file_path: Path) -> List[Result]:
    if not file_path.exists():
        raise InvalidBatchFileError(f"File not found: {file_path}")
    # ... processing
```