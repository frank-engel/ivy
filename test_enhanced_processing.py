#!/usr/bin/env python3
"""
Simple test script for enhanced image processing features.

This script tests the new image processing functions without requiring
the full IVyTools GUI framework.
"""

import cv2
import numpy as np
import os
import sys

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_velocimetry_tools.image_processing_tools import (
    apply_unsharp_mask,
    apply_edge_enhancement,
    apply_difference_of_gaussians,
    apply_bilateral_filter_exposed,
    apply_local_std_dev,
    detect_blur,
    analyze_exposure,
    create_motion_heatmap,
    extract_water_roi_by_color,
)


def create_test_image():
    """Create a simple test image for testing."""
    # Create a gradient image with some texture
    img = np.zeros((480, 640), dtype=np.uint8)

    # Add gradient
    for i in range(480):
        img[i, :] = int(i / 480 * 255)

    # Add some noise for texture
    noise = np.random.randint(0, 30, (480, 640), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Add some edges
    cv2.rectangle(img, (100, 100), (300, 300), 200, 2)
    cv2.circle(img, (450, 240), 80, 150, 3)

    return img


def test_enhancements():
    """Test all enhancement algorithms."""
    print("Testing Enhancement Algorithms...")

    img = create_test_image()

    # Test Unsharp Mask
    print("  - Testing Unsharp Mask...", end=" ")
    try:
        result = apply_unsharp_mask(img, kernel_size=5, sigma=1.0, amount=1.0)
        assert result.shape == img.shape
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test Edge Enhancement
    print("  - Testing Edge Enhancement...", end=" ")
    try:
        result = apply_edge_enhancement(img, alpha=1.5)
        assert result.shape == img.shape
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test Difference of Gaussians
    print("  - Testing Difference of Gaussians...", end=" ")
    try:
        result = apply_difference_of_gaussians(img, sigma1=1.0, sigma2=2.0)
        assert result.shape == img.shape
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test Bilateral Filter
    print("  - Testing Bilateral Filter...", end=" ")
    try:
        result = apply_bilateral_filter_exposed(img, d=9, sigma_color=75, sigma_space=75)
        assert result.shape == img.shape
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test Local Standard Deviation
    print("  - Testing Local Standard Deviation...", end=" ")
    try:
        result = apply_local_std_dev(img, kernel_size=15)
        assert result.shape == img.shape
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_quality_assessment():
    """Test quality assessment functions."""
    print("\nTesting Quality Assessment...")

    img = create_test_image()

    # Test Blur Detection
    print("  - Testing Blur Detection...", end=" ")
    try:
        is_blurry, blur_score = detect_blur(img)
        assert isinstance(is_blurry, (bool, np.bool_))
        assert isinstance(blur_score, (float, np.floating))
        print(f"✓ PASSED (Blur Score: {blur_score:.2f})")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test Exposure Analysis
    print("  - Testing Exposure Analysis...", end=" ")
    try:
        exposure_info = analyze_exposure(img)
        assert 'mean_brightness' in exposure_info
        assert 'is_underexposed' in exposure_info
        assert 'is_overexposed' in exposure_info
        print(f"✓ PASSED (Brightness: {exposure_info['mean_brightness']:.1f})")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_visualization():
    """Test visualization functions."""
    print("\nTesting Visualization Functions...")

    img = create_test_image()

    # Test Motion Heatmap
    print("  - Testing Motion Heatmap...", end=" ")
    try:
        heatmap = create_motion_heatmap(img)
        assert heatmap.shape[2] == 3  # Should be BGR
        assert heatmap.shape[:2] == img.shape
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_roi_extraction():
    """Test ROI extraction functions."""
    print("\nTesting ROI Extraction...")

    # Create a color test image
    img_bgr = cv2.cvtColor(create_test_image(), cv2.COLOR_GRAY2BGR)

    # Make part of it "water-like" (blue-ish)
    img_bgr[100:300, 100:400, 0] = 200  # Blue channel
    img_bgr[100:300, 100:400, 1] = 120  # Green channel
    img_bgr[100:300, 100:400, 2] = 80   # Red channel

    # Test Color-based ROI
    print("  - Testing Color-based ROI Extraction...", end=" ")
    try:
        roi_mask = extract_water_roi_by_color(
            img_bgr,
            color_space='HSV',
            hue_range=(90, 140),
            sat_range=(20, 255),
            val_range=(20, 255)
        )
        assert roi_mask.shape == img_bgr.shape[:2]
        assert roi_mask.dtype == np.uint8
        assert np.all((roi_mask == 0) | (roi_mask == 1))
        print(f"✓ PASSED (ROI pixels: {np.sum(roi_mask)})")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_color_image_handling():
    """Test that functions handle both grayscale and color images."""
    print("\nTesting Color/Grayscale Image Handling...")

    gray_img = create_test_image()
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    functions_to_test = [
        ('Unsharp (gray)', lambda: apply_unsharp_mask(gray_img)),
        ('Unsharp (color)', lambda: apply_unsharp_mask(color_img)),
        ('Edge (gray)', lambda: apply_edge_enhancement(gray_img)),
        ('Edge (color)', lambda: apply_edge_enhancement(color_img)),
        ('DoG (gray)', lambda: apply_difference_of_gaussians(gray_img)),
        ('DoG (color)', lambda: apply_difference_of_gaussians(color_img)),
    ]

    for name, func in functions_to_test:
        print(f"  - Testing {name}...", end=" ")
        try:
            result = func()
            assert result is not None
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Enhanced Image Processing - Functionality Tests")
    print("=" * 60)

    test_enhancements()
    test_quality_assessment()
    test_visualization()
    test_roi_extraction()
    test_color_image_handling()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
