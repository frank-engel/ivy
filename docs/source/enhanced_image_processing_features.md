# Enhanced Image Processing Features for IVyTools

This document describes the new image processing capabilities added to the Image Frame Processing tab.

## Overview

The enhanced image processing features provide:
1. **Water ROI Extraction** - Automatically identify water regions using motion and color
2. **Advanced Image Enhancement** - Multiple algorithms to enhance water surface texture
3. **Frame Quality Assessment** - Detect blur and exposure issues
4. **Visualization Tools** - Motion heatmaps and texture analysis

## 1. Water ROI Extraction

### Temporal Variance Method

Identifies moving water by analyzing pixel variance across frames. Water shows high variance due to motion, while static banks show low variance.

**Usage Example:**
```python
# In your image browser tab instance
variance_map = self.image_browser_tab.compute_water_roi_temporal_variance(sample_rate=5)

# Extract water ROI from variance
roi_mask = self.image_browser_tab.extract_water_roi_auto(
    variance_map=variance_map,
    threshold_percentile=50,  # Higher = more selective
    min_area_percent=5.0      # Minimum region size
)
```

**Parameters:**
- `sample_rate`: Use every Nth frame (default: 5). Higher = faster but less accurate
- `threshold_percentile`: Variance threshold 0-100 (50 = median, 75 = more selective)
- `min_area_percent`: Minimum region area as % of image (removes small noise regions)

### Color-Based Segmentation

Segments water based on color characteristics in HSV or LAB color space.

**Usage Example:**
```python
# Extract water by color (blue-green range)
roi_mask = self.image_browser_tab.extract_water_roi_auto(
    use_color=True,
    hue_range=(90, 140),  # Cyan-blue range in HSV
    threshold_percentile=50
)
```

**HSV Hue Ranges for Different Water Types:**
- Clear blue water: (90, 120)
- Green/turbid water: (40, 90)
- Brown/muddy water: Use LAB color space instead
- Coastal/ocean water: (100, 140)

### Combined ROI Extraction

Combines temporal variance and color segmentation for robust results.

**Usage Example:**
```python
# Compute variance
variance_map = self.image_browser_tab.compute_water_roi_temporal_variance(sample_rate=5)

# Extract using both methods
roi_mask = self.image_browser_tab.extract_water_roi_auto(
    variance_map=variance_map,
    use_color=True,
    hue_range=(90, 140)
)

# Show overlay on image
self.image_browser_tab.show_roi_overlay(roi_mask)
```

## 2. Image Enhancement Algorithms

### Unsharp Masking

Enhances edges and fine details by subtracting a blurred version from the original.

**Best for:** Enhancing water surface texture, making features more visible

**Usage Example:**
```python
# Preview on current frame
self.image_browser_tab.apply_enhancement_to_current_frame(
    'unsharp',
    kernel_size=5,    # Blur kernel size (must be odd)
    sigma=1.0,        # Gaussian sigma
    amount=1.0        # Sharpening strength (1.0 = 100%)
)

# Apply to all frames (see batch processing section below)
```

**Parameters:**
- `kernel_size`: Size of blur kernel (3, 5, 7, 9...). Larger = stronger effect
- `sigma`: Gaussian blur sigma. Larger = more blur, stronger sharpening
- `amount`: Strength multiplier. 0.5 = subtle, 1.0 = normal, 2.0 = aggressive

**Recommended Settings:**
- Fine texture enhancement: `kernel_size=5, sigma=1.0, amount=1.0`
- Aggressive sharpening: `kernel_size=7, sigma=2.0, amount=1.5`

### Edge Enhancement

Highlights edges using Laplacian operator, emphasizing boundaries and patterns.

**Best for:** Highlighting water surface patterns, wave fronts, flow boundaries

**Usage Example:**
```python
self.image_browser_tab.apply_enhancement_to_current_frame(
    'edge',
    alpha=1.5  # Enhancement strength
)
```

**Parameters:**
- `alpha`: Enhancement strength. 1.0 = original, >1.0 = enhanced edges

**Recommended Settings:**
- Subtle edge enhancement: `alpha=1.2`
- Strong edge emphasis: `alpha=2.0`

### Difference of Gaussians (DoG)

Band-pass filter that enhances features at specific scales.

**Best for:** Highlighting specific texture patterns, removing large-scale variations

**Usage Example:**
```python
self.image_browser_tab.apply_enhancement_to_current_frame(
    'dog',
    sigma1=1.0,  # Smaller sigma (fine details)
    sigma2=2.0   # Larger sigma (coarse details)
)
```

**Parameters:**
- `sigma1`: Smaller Gaussian sigma (captures finer details)
- `sigma2`: Larger Gaussian sigma (captures coarser details)
- Difference highlights features between these scales

**Recommended Settings:**
- Fine texture: `sigma1=0.5, sigma2=1.5`
- Medium texture: `sigma1=1.0, sigma2=2.0`
- Coarse patterns: `sigma1=2.0, sigma2=4.0`

### Bilateral Filter

Edge-preserving smoothing that reduces noise while maintaining important boundaries.

**Best for:** Denoising before velocity analysis, smoothing while preserving edges

**Usage Example:**
```python
self.image_browser_tab.apply_enhancement_to_current_frame(
    'bilateral',
    d=9,            # Pixel neighborhood diameter
    sigma_color=75, # Color space sigma
    sigma_space=75  # Coordinate space sigma
)
```

**Parameters:**
- `d`: Diameter of pixel neighborhood (5, 9, 15...)
- `sigma_color`: Filter sigma in color space (higher = more colors mixed)
- `sigma_space`: Filter sigma in coordinate space (higher = larger area)

**Recommended Settings:**
- Light smoothing: `d=5, sigma_color=50, sigma_space=50`
- Strong smoothing: `d=9, sigma_color=100, sigma_space=100`

### Local Standard Deviation

Creates a texture map showing local variation intensity.

**Best for:** Visualizing texture distribution, identifying turbulent regions

**Usage Example:**
```python
self.image_browser_tab.apply_enhancement_to_current_frame(
    'local_std',
    kernel_size=15  # Local neighborhood size
)
```

**Parameters:**
- `kernel_size`: Size of local neighborhood (5, 11, 15, 21...)

**Recommended Settings:**
- Fine-scale texture: `kernel_size=7`
- Medium-scale texture: `kernel_size=15`
- Large-scale patterns: `kernel_size=31`

## 3. Frame Quality Assessment

### Blur Detection

Detects blurry frames using Laplacian variance method.

**Usage Example:**
```python
quality_metrics = self.image_browser_tab.analyze_current_frame_quality()

if quality_metrics:
    is_blurry = quality_metrics['is_blurry']
    blur_score = quality_metrics['blur_score']

    if is_blurry:
        print(f"Warning: Frame is blurry (score: {blur_score:.2f})")
```

**Interpretation:**
- Blur score < 100: Blurry
- Blur score 100-500: Acceptable
- Blur score > 500: Sharp

### Exposure Analysis

Analyzes brightness and contrast to detect under/overexposed frames.

**Usage Example:**
```python
quality_metrics = self.image_browser_tab.analyze_current_frame_quality()

exposure = quality_metrics['exposure']
if exposure['is_underexposed']:
    print("Frame is too dark - consider auto-contrast")
if exposure['is_overexposed']:
    print("Frame is overexposed - clipped highlights")
```

## 4. Visualization Tools

### Motion Heatmap

Visualizes motion intensity across the image using a color map.

**Usage Example:**
```python
# Compute and display motion heatmap
self.image_browser_tab.show_motion_heatmap()

# Or use pre-computed variance
variance_map = self.image_browser_tab.compute_water_roi_temporal_variance()
self.image_browser_tab.show_motion_heatmap(variance_map)
```

**Colors:**
- Blue/Purple: Low motion (static regions)
- Green/Yellow: Medium motion
- Red: High motion (fast-moving water)

### Texture Visualization

Creates color-coded visualization of texture intensity.

**Usage Example:**
```python
# Local standard deviation visualization
self.image_browser_tab.show_texture_visualization(
    method='local_std',
    kernel_size=15
)

# Difference of Gaussians visualization
self.image_browser_tab.show_texture_visualization(
    method='dog',
    sigma1=1.0,
    sigma2=2.0
)

# Edge-based visualization
self.image_browser_tab.show_texture_visualization(
    method='edges',
    alpha=1.5
)
```

### ROI Overlay

Overlays extracted water ROI on the current image.

**Usage Example:**
```python
# Extract and show ROI
roi_mask = self.image_browser_tab.extract_water_roi_auto()
self.image_browser_tab.show_roi_overlay(roi_mask)

# Or let it compute automatically
self.image_browser_tab.show_roi_overlay()  # Auto-computes ROI
```

## 5. Batch Processing

### Applying Enhancements to All Frames

The existing `apply_to_all_frames()` method needs to be extended to support new enhancements. Here's how to update it:

**Updated Method Signature:**
```python
def apply_to_all_frames(self):
    """Apply selected image processing to all frames."""

    # Existing parameters
    do_clahe = self.ivy_framework.checkboxApplyClahe.isChecked()
    do_auto_contrast = self.ivy_framework.checkboxAutoContrast.isChecked()

    # NEW: Add checkboxes/controls for new enhancements
    do_unsharp = self.ivy_framework.checkboxUnsharpMask.isChecked()
    do_edge = self.ivy_framework.checkboxEdgeEnhance.isChecked()
    do_dog = self.ivy_framework.checkboxDoG.isChecked()
    do_bilateral = self.ivy_framework.checkboxBilateral.isChecked()
    do_local_std = self.ivy_framework.checkboxLocalStd.isChecked()

    # Get parameters from GUI controls
    unsharp_params = (
        int(self.ivy_framework.spinboxUnsharpKernel.value()),
        float(self.ivy_framework.doubleSpinboxUnsharpSigma.value()),
        float(self.ivy_framework.doubleSpinboxUnsharpAmount.value())
    )

    # Call ImageProcessor with new parameters
    processing_thread = ImageProcessor()
    processing_thread.progress.connect(self.preprocessor_process_progress)
    processing_thread.finished.connect(self.preprocessor_process_finished)

    processing_thread.preprocess_images(
        image_paths=self.sequence,
        clahe_parameters=clahe_parameters,
        auto_contrast_percent=auto_contrast_percent,
        do_clahe=do_clahe,
        do_auto_contrast=do_auto_contrast,
        # NEW PARAMETERS:
        do_unsharp_mask=do_unsharp,
        unsharp_parameters=unsharp_params,
        do_edge_enhance=do_edge,
        edge_enhance_alpha=edge_alpha,
        do_dog=do_dog,
        dog_parameters=dog_params,
        do_bilateral=do_bilateral,
        bilateral_parameters=bilateral_params,
        do_local_std=do_local_std,
        local_std_kernel=local_std_kernel
    )
```

## 6. GUI Integration Recommendations

### Suggested Tab Layout

```
┌─────────────────────────────────────────────────────┐
│ Frame Processing Tab                                 │
├─────────────────────────────────────────────────────┤
│                                                       │
│ ┌─── Enhancement ───────────────────────────────┐  │
│ │ ☑ CLAHE           Clip: [2.0] Tiles: [8][8]  │  │
│ │ ☑ Auto Contrast   Clip %: [1.0]              │  │
│ │ ☐ Unsharp Mask    Kernel: [5] σ: [1.0] A:[1.0]│  │
│ │ ☐ Edge Enhance    Alpha: [1.5]               │  │
│ │ ☐ DoG Filter      σ1: [1.0] σ2: [2.0]        │  │
│ │ ☐ Bilateral       d:[9] σc:[75] σs:[75]      │  │
│ │ ☐ Local Std Dev   Kernel: [15]               │  │
│ │                                                │  │
│ │ [Apply to This Frame] [Apply to All Frames]   │  │
│ └────────────────────────────────────────────────┘  │
│                                                       │
│ ┌─── Water ROI Extraction ──────────────────────┐  │
│ │ Method: ○ Temporal Variance ○ Color ○ Both    │  │
│ │ Threshold: [50]%  Min Area: [5.0]%            │  │
│ │ Hue Range: [90] - [140]  Sample Rate: [5]     │  │
│ │                                                │  │
│ │ [Compute ROI] [Show Overlay] [Save ROI Mask]  │  │
│ └────────────────────────────────────────────────┘  │
│                                                       │
│ ┌─── Analysis & Visualization ──────────────────┐  │
│ │ [Frame Quality] [Motion Heatmap]              │  │
│ │ [Texture Viz]   [Reset View]                  │  │
│ └────────────────────────────────────────────────┘  │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Connecting GUI Controls (Example)

```python
# In your main GUI initialization (e.g., ivy.py or IVy_GUI.py)

# Connect enhancement preview buttons
self.buttonPreviewUnsharp.clicked.connect(
    lambda: self.image_browser_tab.apply_enhancement_to_current_frame(
        'unsharp',
        kernel_size=self.spinboxUnsharpKernel.value(),
        sigma=self.doubleSpinboxUnsharpSigma.value(),
        amount=self.doubleSpinboxUnsharpAmount.value()
    )
)

# Connect ROI extraction button
self.buttonComputeROI.clicked.connect(
    lambda: self.image_browser_tab.extract_water_roi_auto(
        threshold_percentile=self.spinboxROIThreshold.value(),
        min_area_percent=self.doubleSpinboxMinArea.value(),
        use_color=self.checkboxUseColor.isChecked(),
        hue_range=(self.spinboxHueMin.value(), self.spinboxHueMax.value())
    )
)

# Connect visualization buttons
self.buttonMotionHeatmap.clicked.connect(
    self.image_browser_tab.show_motion_heatmap
)

self.buttonFrameQuality.clicked.connect(
    self.image_browser_tab.analyze_current_frame_quality
)

self.buttonTextureViz.clicked.connect(
    lambda: self.image_browser_tab.show_texture_visualization(
        method='local_std',
        kernel_size=self.spinboxTexKernel.value()
    )
)
```

## 7. Workflow Examples

### Example 1: Basic Enhancement Workflow

```python
# 1. Load images
self.image_browser_tab.open_image_folder()

# 2. Preview enhancement on current frame
self.image_browser_tab.apply_enhancement_to_current_frame(
    'unsharp',
    kernel_size=5,
    sigma=1.0,
    amount=1.5
)

# 3. If satisfied, apply to all frames using the GUI controls
# (User checks the Unsharp Mask checkbox and clicks "Apply to All Frames")
```

### Example 2: Water ROI Extraction Workflow

```python
# 1. Load video frames
self.image_browser_tab.open_image_folder()

# 2. Compute temporal variance (shows progress bar)
variance_map = self.image_browser_tab.compute_water_roi_temporal_variance(sample_rate=5)

# 3. Visualize motion
self.image_browser_tab.show_motion_heatmap(variance_map)

# 4. Extract ROI with appropriate threshold
roi_mask = self.image_browser_tab.extract_water_roi_auto(
    variance_map=variance_map,
    threshold_percentile=60,  # Adjust based on heatmap
    min_area_percent=10.0
)

# 5. Show overlay to verify
self.image_browser_tab.show_roi_overlay(roi_mask)

# 6. Save ROI for later use (implement save functionality)
import numpy as np
np.save('water_roi_mask.npy', roi_mask)
```

### Example 3: Quality Control Workflow

```python
# Analyze all frames for quality issues
for idx, image_path in enumerate(self.image_browser_tab.sequence):
    self.image_browser_tab.sequence_index = idx
    quality = self.image_browser_tab.analyze_current_frame_quality()

    if quality['is_blurry']:
        print(f"Frame {idx}: BLURRY (score: {quality['blur_score']:.1f})")

    if quality['exposure']['is_underexposed']:
        print(f"Frame {idx}: UNDEREXPOSED")

    if quality['exposure']['is_overexposed']:
        print(f"Frame {idx}: OVEREXPOSED")
```

## 8. Performance Considerations

### Computational Cost

**Fast (< 0.1s per frame):**
- Auto Contrast
- Edge Enhancement
- Unsharp Mask

**Medium (0.1-0.5s per frame):**
- CLAHE
- Difference of Gaussians
- Local Standard Deviation

**Slow (0.5-2s per frame):**
- Bilateral Filter

**Batch Operations:**
- Temporal Variance: ~0.1s per sampled frame × num_frames/sample_rate

### Optimization Tips

1. **Use appropriate sample rates:**
   - Temporal variance: sample_rate=5 to 10 for speed
   - Higher sample rates (20+) for very long videos

2. **Process order matters:**
   - Denoising (bilateral) → Enhancement (unsharp) → Quality check
   - Apply heaviest filters first if combining multiple

3. **Preview before batch processing:**
   - Always test on current frame before applying to all

4. **Save intermediate results:**
   - Save variance maps to avoid recomputation
   - Save ROI masks for reuse

## 9. Troubleshooting

### Common Issues

**Issue: Temporal variance shows noise instead of water**
- Solution: Increase sample_rate or check that frames have sufficient motion
- Solution: Lower threshold_percentile

**Issue: Color segmentation doesn't detect water**
- Solution: Use the eyedropper tool to sample actual water color
- Solution: Adjust hue_range based on actual water appearance
- Solution: Try LAB color space for muddy/turbid water

**Issue: ROI includes too much or too little**
- Solution: Adjust threshold_percentile (higher = more selective)
- Solution: Adjust min_area_percent to remove small regions

**Issue: Enhancements make image look worse**
- Solution: Reduce enhancement strength (amount, alpha parameters)
- Solution: Try different enhancement combinations
- Solution: Apply bilateral filter first to denoise

**Issue: Processing is too slow**
- Solution: Increase temporal variance sample_rate
- Solution: Process subset of frames first
- Solution: Reduce bilateral filter neighborhood size

## 10. API Reference Summary

### Main Functions in image_processing_tools.py

**Water ROI:**
- `compute_temporal_variance(image_paths, sample_rate, progress_callback)`
- `extract_water_roi_from_variance(variance_map, threshold_percentile, min_area_percent)`
- `extract_water_roi_by_color(image, color_space, hue_range, sat_range, val_range)`
- `combine_roi_masks(masks, method)`

**Enhancement:**
- `apply_unsharp_mask(image, kernel_size, sigma, amount, threshold)`
- `apply_edge_enhancement(image, alpha)`
- `apply_difference_of_gaussians(image, sigma1, sigma2, normalize)`
- `apply_bilateral_filter_exposed(image, d, sigma_color, sigma_space)`
- `apply_local_std_dev(image, kernel_size)`

**Quality:**
- `detect_blur(image, threshold)`
- `analyze_exposure(image)`

**Visualization:**
- `create_motion_heatmap(variance_map, colormap)`
- `create_texture_visualization(image, method, **kwargs)`
- `overlay_roi_on_image(image, roi_mask, color, alpha)`

### ImageBrowserTab Methods

**ROI Extraction:**
- `compute_water_roi_temporal_variance(sample_rate)`
- `extract_water_roi_auto(variance_map, threshold_percentile, min_area_percent, use_color, hue_range)`

**Quality:**
- `analyze_current_frame_quality()`

**Visualization:**
- `show_motion_heatmap(variance_map)`
- `show_texture_visualization(method, **kwargs)`
- `show_roi_overlay(roi_mask)`

**Enhancement:**
- `apply_enhancement_to_current_frame(enhancement_type, **params)`

---

## Questions or Issues?

For questions about these features or to report issues, please contact the development team or file an issue on the project repository.
