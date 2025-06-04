# Change Log

## [Version v1.0.0.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/1.0.0.0)

**Status**: *Approved*

This is the initial public release of IVyTools. Changes from the previous 
version include:

* The ISO and IVE uncertainty methods have been updated to fix minor 
  errors, and include estimation of uncertainty components due to 
  rectification and the alpha coefficient. Minor tweaks were made to the 
  UI to better label uncertainty parameters. Associated changes were made 
  to the PDF Summary export.
* The documentation has been updated, including a new discussion on how 
  uncertainty is estimated by IVyTools.

## [Version v0.99.0.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.99.0.0)

**Status**: *Testing*

This is the pre-approval version of IVyTools that is being released for 
testing in WFAST and elsewhere ahead of the formal approval of IVyTools at 
a later date. Several changes and improvements to IVyTools have been made 
in this version. Specific highlights include:

* IVy now uses metric units for all backend computations. The numbers 
  displayed in the User Interface are converted to the current display 
  units (for now, hard-coded to English units).
* All angle rotations in the Space-Time Image Velocimetry Tab and 
  Space-Time Image Review tab have been corrected.
* Added the ability to load MKV format videos. 
* The Reporting Tab now attempts to pull metadata from the loaded video and 
  pre-fill the fields.
* Users can now flip the recified image results along the X or Y axis in 
* the orthorectification tab. This solved the "mirror" rectification issues 
  of previous versions.
* IVy now uses a more robust approach to find needed third party binaries 
  (e.g., ffmpeg).
* Many small tweaks to documentation, code docstrings, function efficiency, 
  and more.
* To prepare for full release, changes were made so that a self-installing MSI 
  executable could be made. Notably, IVyTools is now compiled with the 
  `oneDir` approach, meaning the final IVyTools package comprises a folder, 
  rather than a single executable file.

## [Version v0.9.1.1](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.9.1.1)

**Status**: *Do Not Use*

This is an important patch to version 0.9.0.1, correcting issues related to 
Space-Time Image (STI) manual velocities, placement, and calculation of 
search line length. It is highly advised that users upgrade to this current 
version and discontinue the use of v0.9.0.1. 

**Changes:**

* Corrected a bug in the discharge computation function that caused IVy to 
  incorrectly compute the right-edge discharge.
* Modified some of the unit tests associated with testing the midsection 
  discharge computation
* Rewrote the manual STI angle finding functionality and corrected bugs 
  associated with the angle and velocity of manual STI lines


## [Version v0.9.0.1](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.9.0.1)

**Status**: *Do Not Use*

This is an important patch to version 0.9.0.0 correcting issues related to 
vector rotations, Space-Time Image (STI) manual velocities, and discharge 
computations. It is highly advised that users upgrade to this current version 
and discontinue use of v0.9.0.0. 

**Changes:**

* Carefully reviewed all vector math for STIV results. Found and fixed some 
errors in how coordinate system frames were managed (arithmetic vs. 
geographic angle conversions). Modified vector helper functions to ensure 
results are correct across the application.
* Fixed math error in the `component_in_direction` function that was 
incorrectly determining the component velocity in the direction of the 
cross-section normal. Applying manual changes now works as expected.
* Factored the vector plotting code from `open_project` and the stiv 
processor into a new method in `graphics.py`. This change enabled the 
ability to add manually corrected vectors to the STIV image.
* Introduced a global scale method in `quiver` (`graphic.py`), and set a fixed 
scale for all calls in IVy. Now all vectors drawn in the IVy STIV Results 
image will be scaled the same.
* The STIV Results vector color scheme has changed. Now, the two-dimensional 
velocities computed by the STIV algorithm will be plotted as dashed green 
lines. Manually edited Space-Time Images (STI) that produce a new, manual 
velocity, will be plotted as red vectors in the STIV Results image (this 
matches the red streak angle lines, making coloring consistent between tabs). 
The projected normal component velocities remain unchanged.

## [Version v0.9.0.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.9.0.0)

**Status**: *Do Not Use*

After excellent feedback from users, several improvements and new features have
been added to IVy tools in v0.9.0.0. Moreover, version 0.9.0.0 has been 
submitted for USGS Fundamental Science Practices (FSP) review. A summary of 
the most major changes is below.

**Changes:**

* Bugs found with the `*.ivy` project file loading process were fixed, namely
  the GCP, Area-Comp File, and Video loading processes were corrected so that
  the function as expected (see #88).
* Much care was taken to ensure every module, class, and function in the code
  base has documentation. The documentation is viewable in IVy using the help
  menu and the "Code Documentation" link therein.
* Added the `cameratransform` module license to the application.
* Fixed several minor issues. See issues #76, #77, #87, and #90 for more
* The vector conversions responsible for correctly plotting the velocity
  vectors in the image velocimetry tab have been fixed. These vectors are not
  correctly plotted for all cross-section and velocity orientations. See #65.
* Work on enabling of upstream flow (e.g., what may be observed if an eddy is
  within the cross-section) has started. There is new capability to enable
  upstream flow on a per-node basis in the STI Review tab. This feature is
  still experimental, and further improvements are expected.
* The STI Review Tab was overly conservative in filtering valid STIV velocity
  results, turning them to "NaNs" in the discharge results. The filtering has
  been relaxed. Moreover, manually edited STI streak angles previously were
  only retained if they had positive slope. This was excluding valid results,
  and also has been relaxed.
* IVy has a test suite. The tests were revised, and a few new tests added.

## [Version v0.8.2.2](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.8.2.2)

**Status**: *Do Not Use*

This is a new release for testing several new features. The biggest and most
noticeable change is that the application has been renamed to Image Velocimetry
Tools (IVy Tools). This change was suggested by several users and leadership,
and better describes the purpose and nature of the application.

In addition, several new reporting and review features have been added. Work
has been completes to address reported bugs and issues, improve stability, and
performance.

**Changes:**

* The IVy project file now includes everything needed to process velocity and
  compute discharge for the given project, excluding the video file. However,
  if the video is located in the same folder as the `*.ivy` file, IVy will load
  the video. This means that the `*.ivy` file can be shared with others without
  the need to share other files or worry about path errors.
* IVy now has a Check For Updates feature, available in the About menu.
* When users save the IVy project file, if a discharge result exists IVy will
  write a system comment to the project with a timestamp, current values, and
  username. In this way, the processing of changes is tracked by the project.
* When adding comments with the Comment Dialog button, IVy will automatically
  select the comment category based on which tab is currently active in the
  application.
* The STI Review Tab table now includes a Comments column. This enables users
  to add comments to a particular STI result and provide context to reviewers
  of the project. This comment field is capped at 240 characters. The comments
  are saved and loaded to and from the IVy project file, and also included in
  the PDF Summary export.
* The Documentation has been updated with new descriptions of features and
  their implementation.
* Several other "behind the scenes" changes have been made to increase code
  stability and readability.
* Patch-level semantic versioning added. 

## [Version v0.8.2](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.8.2)

**Status**: *Do Not Use*

This is a minor release that corrects some issues with the previous version and
adds a few handy features based on users' testing and feedback.

**Changes:**

* A glossary of terms has been added to the documentation
* Manual STI streak angles are now saved in the project session file and are
  reloaded when a user opens a new project
* The PDF report has been updated to include the STIV parameters and Project
  Description text
* The PDF report now also incorporates the STIs into the PDF. The STIs are
  embedded in a table, looking much like they do in the IVy Space-Time Image
  Results tab.

## [Version v0.8.1](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.8.1)

**Status**: *Do Not Use*

A new release to fix several small errors and annoyances. More rubust reporting
and error handling, plus more. Users should not run v0.8.0, and should upgrade
to v0.8.1 for the best experience.

**Changes:**

* IVy now provides users the capability to review and manually adjust
  Space-Time Image streak angles. Manually adjusted velocities can then be used
  to recompute a revised discharge estimate.
* The current video file is now shown in the IVy Window Title.
* Error handling has been modified to match the approach used in other USGS
  software products like QRev and RIVRS. Errors are now shown in a dialog
  message box for users. They are also logged to the
  user's `AppData/Roaming/IVy` directory.
* Tweaks to the Summary PDF Report made to make the report more useful.
* STIV and Discharge results are now written to CSV files in the project swap,
  and also are saved in the `*.ivy` project session file.
* Continued improvements to code documentation, users guide, and general code
  readability.

## [Version v0.8.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.8.0)

**Status**: *Do Not Use*

This release marks a major improvement to IVy since the initial alpha release.

**Changes:**

* A bug associated with computing the "wetted top width" of the cross-section
  geometry has been fixed. IVy now correctly computes the edge of water
  locations for the wetted cross-section. These edge of water locations are
  used to correctly scale the cross-section geometry to the image pixel
  locations provided in the Cross-Section Geometry tab. Although in most cases,
  discharge values from version 0.7.0 will be unchanged, it is possible that
  this fix will impact past results. IVy session files saved in version 0.7.0
  can be loaded and reprocessed in version 0.8.0 if necessary.
  See [issue 68](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/issues/68)
  for more information.
* The Reporting Tab gets a complete overhaul and massive improvement.
    * A Comment Engine has been added, allowing users to add text comments,
      categorized by processing step at any point. These comments are collected
      and displayed in the Application Comments table in the Reporting tab,
      saved in the IVy project file, and eincluded in the revised Summary PDF
      report.
    * Standardized Discharge Measurement metadata fields are added to the
      Reporting Tab. Some validation is also performed.
    * A substantially revised Summary PDF report has been added. This PDF
      report can be exported once at the Discharge Computation processing step.
      The PDF provides both at-a-glance and detailed information suitable for
      work, check, review, and archival processes.
* Fixed a bug that did not correctly apply changed to alpha in the Discharge
  Tab.
* Fixed Discharge Table formatting
* Fixed Grid creation point numbering so that they match the Discharge Station
  numbers
* Added a persistent Toolbar that provides quick access to comment file
  management, help, and comment functionality
* Revised and improved the User Guide
* Removed binary dependencies that are no longer used by IVy, shrinking the
  overall EXE file size
* Many other small code fixes and tweaks

## [Version v0.7.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.7.0)

**Status**: *Do Not Use*

This release is the alpha version of the complete IVy Framework Application.

**Changes:**

* A documentation browser has been added, giving users a comprehensive overview
  of all major items in the IVy Framework application. Although this is not a
  technical manual, users with an essential background in image velocimetry
  techniques should be able to use IVy based on this new documentation.
* IVy will now use the normal velocity in the Discharge midsection computations
  rather than the 2D velocity magnitude.
* The discharge plots have been updated. Users can now interactively select
  stations from the plots and see the results in the Discharge Stations table.
* Although other gridding methods are included in IVy, they have been disabled
  for this release. Currently, IVy will only function in the "Cross-section
  Discharge" mode.
* The PDF Summary Report functionality has been improved.
* IVy units are fixed to English only for now.
* Added significantly to code documentation and unit tests.
* Major refactoring of the app to facilitate Sphinx documentation and conform
  to other USGS Python app norms (QRev, SVT, AreaComp, etc.).
* Many other minor improvements.

## [Version v0.6.9](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.6.9)

**Status**: *Do Not Use*

This is the Phase 5 development release.

**Changes:**

* Added initial version of the Discharge Results plots.

## [Version v0.6.8](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.6.8)

**Status**: *Do Not Use*

This is the Phase 5 development release.

**Changes:**

* Hardened the buttons and functions so that it is more difficult to crash IVy.
  When required inputs are needed, the app now prompts the user wit ha dialog
  and instructions.
* Disabled functionality related to STIV-Optimization mode for the time being
* Drastically improved the speed and stability of the Image Stack process
* Improved the stability of the homography and camera matrix rectification
  methods
* Locked out Metric units in this release. All units in IVy will be presented
  in English units for now. All units conversions of input data (e.g. GCP or
  cross-section geometry) will be converted in the back end to English units.
  This functionality has been thoroughly tested to ensure it is correct.
* Drawing the cross-section point locations in the Cross-section tab will be
  propagated to the Grid Generation tab.

## [Version v0.6.6](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.6.6)

**Status**: *Do Not Use*

This is the Phase 5 development release.

**Changes:**

* Cross-section Geometry support added. AreaComp3 has been implemented within
  IVy to manage cross-section information. Just load a saved AreaComp3 MAT-file
  to import geometry.
* Gridding gets a major revamp. Users can now create grid points using four
  methods (single point, simple line, cross-section line, or grid). The
  resulting grid is used for image velocimetry.
* The image stack process has been greatly improved. As a preprocessing step
  for image velocimetry, an “image stack” has to be created. The process of
  creating stacks has been completely refactored to be more stable and faster.
* The STIV Exhaustive mode now includes visualizing the search line parameters
  in the STIV tab image, which greatly improves parameter choice.
* STIV Exhaustive also now supports a Gaussian blur preprocessor. In some
  cases, applying a blur may improve STIV results.
* STIV Exhaustive can now save Space Time Images (STIs). When enabled, the STIs
  are saved in the “1-Images” folder of the Project Manager. This means the
  Image Preprocessing Tab viewer can view them by setting the filter to “STI*
  .jpg”
* A new Discharge Computation tab has been added. If a cross-section is loaded,
  IVy will now process STIV vector velocities into discharge. IVy uses the
  Midsection method described by Turnipseed and Sauer (2010). Additionally,
  midsection uncertainty is estimated using the ISO 648 and IVE methods. Users
  can rate the measurement.

## [Version v0.3.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.3.0)

**Status**: *Do Not Use*

This is the Phase 3 alpha release.

**Changes:**

* Two implementations of STIV have been added: Exhaustive and Optimized. Each
  will produce a two-dimensional velocity field at nodes defined in a grid.
* A grid creation tab has been added. The grid creator is to produce a "
  results" grid, where velocity output will be saved to
* The STIV algorithms produce a results grid CSV file that can be saved
  independently or within the ivy project structure. They also produce an image
  for viewing the results. As this is the initial implementation, that image is
  not the final form of the results output.
* The process of separating a dataclass containing all the input and output
  class instance variables for the IVy Framework has been started.
* Additionally, I've started creating a "Base Image Velocimetry" class that is
  meant to be a framework for adding new methods. This base class would then be
  available to other developers to subclass and overload methods. In this
  manner, I can specify and share the required input and outputs IVy needs to
  operate, and the developer only needs to ensure that their results fit into
  that framework.

---

## [Version v0.2.0](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases/0.2.0)

**Status**: *Do Not Use*

This is the Phase 2 alpha release.

**Changes:**

* A new project structure and file management system has been created. Now IVy
  uses the OS native temporary file management to orgnaize the project
  structure.
  All images ae now just saved as `f*%05d.jpg` files.
* The new file management also adds a "Tree" view Project Hierarchy tool to
  relevant IVy tabs. Now you can see how and what files are already present in
  the project. This will help ensure the needed inputs for further processing
  steps are selected.
* A new project file configuration has been added. This new project file is
  saved when users save a project. The file (`*.ivy`) contains the current
  application configuration, and all processing files (excluding the input
  video).
* There is now a Space-Time Image Velocimetry module. It currently runs, but is
  not connected and accessible to the User Interface at this time.
* Several unit tests have been made, enabling faster development of new
  features. Test coverage is far from complete, however.

---

## Software and Firmware Status Definitions

- **Required Minimum**: Minimum version required. This version has proven
  stable
  and may contain enhancements that are significant over previously required
  versions.
- **Recommended**: Shown to have been reliable and contains features that
  result
  in a recommended upgrade over the required version. There could be a few
  specific use cases where this version may have issues that would result in
  some
  users not using this version. If so, those cases will be noted.
- **Allowed**: Deemed reliable during initial testing. Any issues will be
  noted
  along with improvements available over prior versions. Use of allowed
  versions
  may be desired when the changes benefit many of the
  user's conditions or equipment. For example, a new software version is
  released that adds support for new hardware. If the user has this hardware,
  they must upgrade to the more recent software before it becomes recommended
  or required. The use of these versions by experienced users will also help in
  identifying any unknown issues.
- **Testing**: Software currently in testing; any known issues or advantages
  over
  prior release will be noted. Using a version that is in testing should
  usually be limited to advanced users that can troubleshoot potential issues
  and provide feedback on any irregularities or problems observed.
- **Do Not Use**: A version either before the required minimum or that contains
  issues that significantly affect operations.

**Note**: A version may remain in **Allowed** or **Testing** indefinitely.
Example: A new version is released while the prior version is still in testing.
In this case, the prior version may remain in testing while future testing
efforts are placed on the newer version.