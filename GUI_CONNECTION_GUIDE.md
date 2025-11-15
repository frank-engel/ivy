# GUI Connection Guide for Enhanced Image Processing Features

## Overview

The UI elements have been added to `IVy_GUI.ui`. After running `ui2py.bat`, you'll need to connect the new buttons and checkboxes to the backend functions.

## Step 1: Run ui2py.bat

```bash
ui2py.bat
```

This will regenerate `IVy_GUI.py` from the .ui file with all the new controls.

## Step 2: Add Signal Connections

In your main IVy class (likely in `ivy.py` or where you initialize the GUI), add these connections in your `__init__` or setup method:

```python
def setup_image_processing_connections(self):
    """Connect image processing tab buttons and controls to their handlers."""

    # ====================================================================
    # Enhancement Checkboxes (for batch processing)
    # ====================================================================
    # These are used when user clicks "Apply to All Frames"
    # The existing apply_to_all_frames method will need to check these

    # ====================================================================
    # Water ROI Buttons
    # ====================================================================

    # Compute Temporal Variance
    self.buttonComputeTemporalVariance.clicked.connect(
        lambda: self._compute_and_cache_variance()
    )

    # Extract Water ROI
    self.buttonExtractWaterROI.clicked.connect(
        lambda: self._extract_roi()
    )

    # Show ROI Overlay
    self.buttonShowROIOverlay.clicked.connect(
        lambda: self.image_browser_tab.show_roi_overlay()
    )

    # ====================================================================
    # Analysis & Visualization Buttons
    # ====================================================================

    # Frame Quality Analysis
    self.buttonFrameQuality.clicked.connect(
        self.image_browser_tab.analyze_current_frame_quality
    )

    # Motion Heatmap
    self.buttonMotionHeatmap.clicked.connect(
        lambda: self._show_motion_heatmap()
    )

    # Texture Visualization
    self.buttonTextureViz.clicked.connect(
        lambda: self._show_texture_dialog()
    )

    # Reset View
    self.buttonResetView.clicked.connect(
        lambda: self._reset_image_view()
    )


def _compute_and_cache_variance(self):
    """Compute temporal variance and cache for reuse."""
    try:
        sample_rate = int(self.lineeditSampleRate.text())
        self._cached_variance_map = self.image_browser_tab.compute_water_roi_temporal_variance(
            sample_rate=sample_rate
        )
    except ValueError:
        QtWidgets.QMessageBox.warning(
            self,
            "Invalid Input",
            "Sample rate must be an integer.",
            QtWidgets.QMessageBox.Ok
        )


def _extract_roi(self):
    """Extract water ROI using current settings."""
    try:
        threshold = float(self.lineeditROIThreshold.text())

        # Use cached variance if available, otherwise compute it
        if not hasattr(self, '_cached_variance_map') or self._cached_variance_map is None:
            self._compute_and_cache_variance()

        if self._cached_variance_map is not None:
            self._cached_roi_mask = self.image_browser_tab.extract_water_roi_auto(
                variance_map=self._cached_variance_map,
                threshold_percentile=threshold,
                min_area_percent=5.0
            )
    except ValueError:
        QtWidgets.QMessageBox.warning(
            self,
            "Invalid Input",
            "Threshold must be a number between 0 and 100.",
            QtWidgets.QMessageBox.Ok
        )


def _show_motion_heatmap(self):
    """Show motion heatmap using cached or computed variance."""
    if not hasattr(self, '_cached_variance_map') or self._cached_variance_map is None:
        self._compute_and_cache_variance()

    if self._cached_variance_map is not None:
        self.image_browser_tab.show_motion_heatmap(self._cached_variance_map)


def _show_texture_dialog(self):
    """Show dialog to choose texture visualization method."""
    # Simple implementation - you can make this fancier with a dialog
    methods = ['local_std', 'dog', 'edges']

    method, ok = QtWidgets.QInputDialog.getItem(
        self,
        "Texture Visualization",
        "Select visualization method:",
        methods,
        0,
        False
    )

    if ok:
        self.image_browser_tab.show_texture_visualization(method=method)


def _reset_image_view(self):
    """Reset image view to original frame."""
    # Reload current image
    if self.image_browser_tab.sequence and self.image_browser_tab.sequence_index < len(self.image_browser_tab.sequence):
        current_path = self.image_browser_tab.sequence[self.image_browser_tab.sequence_index]
        self.image_browser_tab.image = QtGui.QImage(current_path)
        self.image_browser_tab.imageBrowser.scene.setImage(self.image_browser_tab.image)
        self.update_statusbar("IMAGE BROWSER: View reset to original image")
```

## Step 3: Update apply_to_all_frames Method

Modify the existing `apply_to_all_frames` method in `image_browser.py` to include the new enhancements:

```python
def apply_to_all_frames(self):
    """Apply selected image processing to all frames."""

    message = "IMAGE BROWSER: Applying image preprocessing to all frames..."
    self.ivy_framework.update_statusbar(message)
    self.ivy_framework.progressBar.show()

    # Existing parameters
    do_clahe = self.ivy_framework.checkboxApplyClahe.isChecked()
    do_auto_contrast = self.ivy_framework.checkboxAutoContrast.isChecked()
    clip = float(self.ivy_framework.lineeditClaheClipLimit.text())
    horz_tile_size = int(self.ivy_framework.lineeditClaheHorzTileSize.text())
    vert_tile_size = int(self.ivy_framework.lineeditClaheVertTileSize.text())
    clahe_parameters = (clip, horz_tile_size, vert_tile_size)
    auto_contrast_percent = float(self.ivy_framework.lineeditAutoContrastPercentClip.text())

    # NEW: Get enhancement parameters
    do_unsharp = self.ivy_framework.checkboxUnsharpMask.isChecked()
    do_edge = self.ivy_framework.checkboxEdgeEnhancement.isChecked()
    do_dog = self.ivy_framework.checkboxDoG.isChecked()
    do_bilateral = self.ivy_framework.checkboxBilateral.isChecked()
    do_local_std = self.ivy_framework.checkboxLocalStd.isChecked()

    # Get unsharp mask parameters
    if do_unsharp:
        try:
            unsharp_kernel = int(self.ivy_framework.lineeditUnsharpKernel.text())
            unsharp_sigma = float(self.ivy_framework.lineeditUnsharpSigma.text())
            unsharp_amount = float(self.ivy_framework.lineeditUnsharpAmount.text())
            unsharp_parameters = (unsharp_kernel, unsharp_sigma, unsharp_amount)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self.ivy_framework,
                "Invalid Parameters",
                "Please check unsharp mask parameters.",
                QtWidgets.QMessageBox.Ok
            )
            self.ivy_framework.progressBar.hide()
            return
    else:
        unsharp_parameters = (5, 1.0, 1.0)  # defaults

    # For other enhancements, use defaults (or add more controls if needed)
    edge_alpha = 1.5
    dog_parameters = (1.0, 2.0)
    bilateral_parameters = (9, 75, 75)
    local_std_kernel = 15

    # Create processor
    processing_thread = ImageProcessor()
    processing_thread.progress.connect(self.preprocessor_process_progress)
    processing_thread.finished.connect(self.preprocessor_process_finished)

    # Process images with all parameters
    self.ivy_framework.progressBar.setValue(0)
    self.ivy_framework.progressBar.show()

    processing_thread.preprocess_images(
        image_paths=self.sequence,
        clahe_parameters=clahe_parameters,
        auto_contrast_percent=auto_contrast_percent,
        do_clahe=do_clahe,
        do_auto_contrast=do_auto_contrast,
        # NEW PARAMETERS:
        do_unsharp_mask=do_unsharp,
        unsharp_parameters=unsharp_parameters,
        do_edge_enhance=do_edge,
        edge_enhance_alpha=edge_alpha,
        do_dog=do_dog,
        dog_parameters=dog_parameters,
        do_bilateral=do_bilateral,
        bilateral_parameters=bilateral_parameters,
        do_local_std=do_local_std,
        local_std_kernel=local_std_kernel
    )
```

## Step 4: Add Preview Functionality for Single Frame

Update the `apply_to_this_frame` method to support enhancement previews:

```python
def apply_to_this_frame(self):
    """Applies selected processing to the current frame for preview."""

    if not self.sequence:
        return

    current_image = self.sequence[self.sequence_index]
    cv_image = image_file_to_opencv_image(current_image)
    height, width = cv_image.shape[:2] if len(cv_image.shape) == 2 else cv_image.shape[:2]

    # Apply existing enhancements (CLAHE, Auto Contrast)
    if self.ivy_framework.checkboxApplyClahe.isChecked():
        clip = float(self.ivy_framework.lineeditClaheClipLimit.text())
        horz = int(self.ivy_framework.lineeditClaheHorzTileSize.text())
        vert = int(self.ivy_framework.lineeditClaheVertTileSize.text())
        cv_image = apply_clahe_to_image(cv_image, clip_size=clip,
                                        horz_tile_size=horz, vert_tile_size=vert)

    if self.ivy_framework.checkboxAutoContrast.isChecked():
        percent = float(self.ivy_framework.lineeditAutoContrastPercentClip.text())
        cv_image, alpha, beta = automatic_brightness_and_contrast_adjustment(
            cv_image, clip_histogram_percentage=percent
        )

    # NEW: Apply enhancement filters
    if self.ivy_framework.checkboxUnsharpMask.isChecked():
        try:
            kernel = int(self.ivy_framework.lineeditUnsharpKernel.text())
            sigma = float(self.ivy_framework.lineeditUnsharpSigma.text())
            amount = float(self.ivy_framework.lineeditUnsharpAmount.text())
            cv_image = apply_unsharp_mask(cv_image, kernel_size=kernel,
                                          sigma=sigma, amount=amount)
        except ValueError:
            pass

    if self.ivy_framework.checkboxEdgeEnhancement.isChecked():
        cv_image = apply_edge_enhancement(cv_image, alpha=1.5)

    if self.ivy_framework.checkboxDoG.isChecked():
        cv_image = apply_difference_of_gaussians(cv_image, sigma1=1.0, sigma2=2.0)

    if self.ivy_framework.checkboxBilateral.isChecked():
        cv_image = apply_bilateral_filter_exposed(cv_image, d=9,
                                                   sigma_color=75, sigma_space=75)

    if self.ivy_framework.checkboxLocalStd.isChecked():
        cv_image = apply_local_std_dev(cv_image, kernel_size=15)

    # Display result
    pixmap = convert_opencv_image_to_qt_pixmap(cv_image,
                                                display_width=width, display_height=height)
    self.imageBrowser.scene.setImage(pixmap)

    message = f"IMAGE BROWSER: Applied processing to current frame"
    self.ivy_framework.update_statusbar(message)
```

## Step 5: Initialize Cache Variables

In your main GUI class `__init__`:

```python
def __init__(self):
    # ... existing init code ...

    # Initialize cache for ROI processing
    self._cached_variance_map = None
    self._cached_roi_mask = None

    # Connect the new controls
    self.setup_image_processing_connections()
```

## Summary of New UI Elements

### Enhancement Section
- `checkboxUnsharpMask` - Enable unsharp masking
- `lineeditUnsharpKernel` - Kernel size (K)
- `lineeditUnsharpSigma` - Gaussian sigma (σ)
- `lineeditUnsharpAmount` - Sharpening amount (A)
- `checkboxEdgeEnhancement` - Enable edge enhancement
- `checkboxDoG` - Enable Difference of Gaussians
- `checkboxBilateral` - Enable bilateral filtering
- `checkboxLocalStd` - Enable local std dev

### Water ROI Section
- `buttonComputeTemporalVariance` - Compute variance map
- `buttonExtractWaterROI` - Extract water ROI
- `buttonShowROIOverlay` - Show ROI overlay
- `lineeditROIThreshold` - Variance threshold percentile
- `lineeditSampleRate` - Frame sampling rate

### Analysis & Viz Section
- `buttonFrameQuality` - Analyze frame quality
- `buttonMotionHeatmap` - Show motion heatmap
- `buttonTextureViz` - Show texture visualization
- `buttonResetView` - Reset to original view

## Testing

After implementing these connections:

1. Run `ui2py.bat` to regenerate the GUI Python file
2. Start IVyTools
3. Load a sequence of images
4. Test each button/checkbox individually
5. Try combining multiple enhancements
6. Test the ROI extraction workflow:
   - Compute Variance → Show Motion Heatmap → Extract ROI → Show Overlay

## Notes

- The Enhancement checkboxes work with both "Apply to This Frame" (preview) and "Apply to All Frames" (batch)
- Water ROI buttons work on the current sequence of loaded images
- The variance map is cached to avoid recomputing for multiple operations
- Reset View allows users to return to the original image after visualizations
