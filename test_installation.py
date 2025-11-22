#!/usr/bin/env python3
"""Test script to validate package installation.

Run this after installing to verify everything works:
    pip install .
    python test_installation.py
"""

import sys
import importlib

def test_import(module_name, description):
    """Test importing a module."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        return False

def main():
    """Run installation tests."""
    print("=" * 60)
    print("Testing image_velocimetry_tools installation")
    print("=" * 60)

    tests = [
        # Core modules
        ("image_velocimetry_tools", "Core package"),
        ("image_velocimetry_tools.gui", "GUI components"),
        ("image_velocimetry_tools.batch", "Batch processing"),
        ("image_velocimetry_tools.batch.models", "Batch models"),
        ("image_velocimetry_tools.services", "Services layer"),
        ("image_velocimetry_tools.api", "Public API"),

        # API components
        ("image_velocimetry_tools.api.batch_api", "Batch API"),

        # Key dependencies
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("cv2", "OpenCV"),
        ("skimage", "scikit-image"),
        ("PyQt5", "PyQt5"),
        ("h5py", "HDF5 support"),
    ]

    print("\nImport Tests:")
    print("-" * 60)
    results = [test_import(module, desc) for module, desc in tests]

    # Test API can be imported
    print("\nAPI Tests:")
    print("-" * 60)
    try:
        from image_velocimetry_tools.api import run_batch_processing, BatchResults, JobResult
        print("✓ API imports successful")
        print(f"  - run_batch_processing: {callable(run_batch_processing)}")
        print(f"  - BatchResults: {BatchResults is not None}")
        print(f"  - JobResult: {JobResult is not None}")
        api_ok = True
    except Exception as e:
        print(f"✗ API imports failed: {e}")
        api_ok = False
        results.append(False)

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    failed = total - passed

    if api_ok:
        passed += 1
        total += 1
    else:
        total += 1

    print(f"Results: {passed}/{total} tests passed")

    if failed == 0:
        print("✓ All tests passed! Installation successful.")
        return 0
    else:
        print(f"✗ {failed} test(s) failed. Check missing dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
