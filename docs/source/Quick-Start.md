# Quick Start

When you start Ivy Framework, the application will start on the **Video Pre-processing** tab. The basic workflow for using IVy Framework is:
1. Load video containing candidate image velocimetry data
2. Use the pre-processing functions to select, prepare, and export the desired image frames for further analysis
3. Establish ground control point (GCP) correspondences between the image and the real world
4. Rectify a single frame to compute transformation parameters and evaluate errors
5. Apply transformation to all relevant image frames to produce images ready for image velocimetry analysis
6. Save the project often so that you can return to the same point using the *Load Project* functionality

## Load video
The easiest way to load a video is to drag and drop the file into the video viewer in the **Video Pre-processing** tab. You can also select *Import Video (Ctrl+V)* from the *Import* file menu

![import-vid](../assets/import-vid.png)

## Pre-process video
Select the desired start and end point in the video using the video playhead. Click the *Clip Start* and *Clip End* buttons to set the clip timings. 

There are a variety or pre-processing options for changing contrast, color, luma, and more. These are not discussed in detail here in the Quick Guide. Settings chosen here will be respected when the video is processed.

If desired, a video clip (saved as an MPEG-4 video file) can be created with the current settings with the *Create Video Clip* button. 

## Frame preparation
Here, the user can activate motion stabilization and/or lens distortion correction. 

### Image stabilization
If activated, the IVy framework will perform a 2-pass image stabilization routine which does not need masking of the water surface. In general, the results of this method have been satisfactory for shaky handheld video and drone imagery. The results are less satisfactory for video captured by a drone with highly instable vertical movements. 

### Lens distortion correction
If activated, the IVy Framework will perform a simplified `k1, k2` radial distortion removal based on supplied lens characteristics. To use, check the *Correct radial lens distortion?* check box, and click the *Load Lens Characteristics* button. A new window will open where you can supply estimated  `k1, k2` parameters and the principal point.

## Frame extraction
Once satisfied with the video pre-processing settings, specify the frame step desired. A frame step of 1 would take every video frame in a loaded video, whereas a frame step of 5 would only take every 5th frame. When the frame step is changes (tip: press enter after editing the number), the extraction parameters will be updated.

Pressing the *Extract Video Frames* button will cause the application to start exporting frames according to current settings. 

By convention, the frames will be saved to a folder having the same name as the loaded video, with suffixes according the the current settings. All frames extracted during this step will have a naming convention of `f*.jpg` if no stabilization is requested, and `s*.jpg` if stabilization is requested. For example frame 1 would be named `f00001.jpg`. The `f` denotes "extracted frame" and `s` denotes "motion stabilized frame" respectively. If stabilization is requested, the IVy Framework will write both the original and stabilized frames to the output folder.

## Use of the **Image Frame Processing** tab
By default, if frames were extracted in the **Video Pre-processing** tab, they will be automatically loaded in the **Image Frame Processing** tab. This tab works like an image browser and basic image editor. It will display sequences of images based on the *Frame filtering* string. This box expects strings which help filter lists of similarly named files. For example, `f*.jpg` would match any frame exported in the steps above. If this box is edited, click the *Apply File Filter* button to update results. 

The image browser will show the current frame. This is an interactive image view. The following mouse events are supported (when the mouse cursor is in the image):
* Hold left mouse button and drag to select a zoom region and zoom in
* Right click to zoom to last extent (undo zoom)
* Use the mouse scroll wheel to zoom in and out
* When zoomed in, press in the mouse wheel and hold  (if equipped) to pan

Clicking the forward and back arrows to the left side of the image browser will display the next and previous matching image in the filtered sequence respectively. Note that *Ctrl+Right Arrow Key* and *Ctrl+Left Arrow Key* will also change the current image.  The path to the current image will be shown in the status bar at the bottom of the application.

... Processing capabilities to be added/discussed TBD

## Orthorectification
In the **Orthorectification** tab, users can transform images with perspective into rectified maps. The first step is to import a GCP image using the *Import Ground Control Image (Ctrl+G)* option in the *Import* file menu. Typically, this will be an annotated image showing the location of the various GCP in the scene which were identified and surveyed in the field. *Note: this image needs to be the exact same dimensions as the loaded extracted frames.*

### Digitizing GCP
To transform the perspective imagery there needs to be known point correspondences between the real world GCP coordinates and the pixel location of those coordinates in the loaded GCP image. Three methods are supported:
1. Two point scaling: this method uses the known ground distances between 2 GCP to calculate the *pixel ground scale distance (Pixel GSD)*
2. Homography: this method requires the X,Y,Z real world coordinates of 4 GCP located **on the plane of the water surface** to compute the homography transform between the image camera and water surface.
3. Full camera matrix solution: this method requires the X,Y,Z real world coordinates of at least 6 GCP located in the field of view of the camera image to compute the camera *projective matrix* which relates the image and real world coordinate spaces. To create a transformed/rectified image, the elevation of the water surface in the same coordinate system as the GCP real world points is required.

To make these correspondences, users will digitize points on the image in conjunction with the *Points Table*. The recommended workflow is to:
1. Load a GCP image if not already loaded
2. Add rows to the *Points Table* using the *Plus button* located to the left of the *Points Table*
3. Edit the table to name the points (the *# ID* column). To do this, just click into the cell of the table and type. The enter key or tab key will set the entered value
4. Edit the *Use in Rectification* column in the *Points Table* to ensure any digitized point is drawn on the current GCP image. The application expects a "truthy" value in this column (e.g. "Yes", "no", "true", "False", "y", "F" all will work, the field is not case sensitive). 
5. Digitize the GCP pixel coordinates for each point. To do this, press the "crosshairs" button to the left of the *Points Table*. This button will toggle on/off. When toggled on (the button turns blue), the application is in point digitization mode. 
	1. Select a row in the *Points Table* by clicking on it. This will be the point which will be digitized
	2. Left click on the GCP image to set the GCP point in the image. As long as the *Use in Rectification* entry for the current point is a true "truthy" value, the point will immediately be drawn in the image where the user clicked.
	3. The image zoom and pan functions are still active. Refine the GCP point location.
	4. The pixel coordinates of the current point are shown in the *Points Table*
6. Select the next point row and repeat the above steps to digitize the GCP image pixel coordinates. Repeat for all GCP points. 
7. If needed, supply the known water surface elevation

### Rectification
Once the GCP *Points Table* is completed, click the *Rectify Current Image* button to perform an initial calibration of the rectification results (the method will automatically selected based on the number of points where *Use in Rectification* is true). The application will present the rectified image in the image view frame to the right. Also, the reprojection errors will be added to the *Points Table* along with the *Pixel GSD* and overall RMSE of the rectification. 

Evaluate the results for accuracy by considering the point reprojection errors. For the full camera matrix method, it is often good practice to turn on or off certain GCP with higher error to determine if a better fit can be made. 

### Export projected frames
Once satisfied with the rectification results of the single frame, the transformation can be applied to all specified frames by clicking the *Export Projected Frames* button. The process will convert all frames currently loaded in the **Image Frame Processing** tab with the specific transformation parameters. When the button is clicked, a message box will open asking the user to verify the first frame to be transformed. *Note: check this frame carefully before exporting all frames to ensure proper results.*