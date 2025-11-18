<!-- Last update: 11/17/2025 18:47 -->

[![python](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<!--[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1126/science.1058040)](https://juleskreuer.eu/projekte/citation-badge/)
-->

# <img src="image_velocimetry_tools/gui/icons/IVy_Logo.svg"  width="120" height="120">Image Velocimetry Tools (IVy Tools)

The IVyTools is a Python application which provides an end-to-end workflow 
support for the processing of video and imagery into water velocity and 
streamflow measurements. IVyTools uses Space-Time Image Velocimetry to 
compute water surface velocity from sequences of images.

The latest release can be downloaded 
[here](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/releases).

The change log and links to releases can be viewed 
[here](/docs/source/changelog.md).

Bugs and enhancement requests can be reported 
[here](https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy/-/issues).

Details on how to contribute can be viewed [here](/CONTRIBUTING.md).



## Authors and Acknowledgments
**Suggested Citation:**
Engel, F. L., and Knight, T. 2025. Image Velocimetry Tools
(IVyTools): U.S. Geological Software Release, 
https://doi.org/10.5066/P1KMVCNY.

### Development Team
* **Frank L. Engel, Ph.D.** (ideation, primary development)
* **Travis Knight** (AreaComp, cross-section engine)
* Carl Legleiter (STIV Core, PIV Core)
* Chris MacPherson (Testing, Review)
* Paul Kinzel (Testing)
* Matt Gyves (Testing)
* John Fulton (Testing)
* Russ Lotspeich (Testing)
* Students in the U.S. Geological Survey (USGS), Water Mission Area 
  Hydrologic—Remote Sensing Branch sponsored Non-contact classes  (Testing)

## Installation

1. IVyTools will be available to USGS users through WFAST to 
   registered computers. Internal USGS users with computers not registered in 
   WFAST can download the software from the IVyTools repository at the link 
   in the table above.
2. No installation is required to use the software, however the LAVFilters 
   codec is required to enable video playback. See the Installation 
   instructions in the Release Notes for more information.
3. To run the application, extract the contents from the downloaded `.zip` file 
   into a folder of the user's choice.
4. The `.zip` file includes a user guide that contains detailed instructions on 
   IVyTools's features and an executable to run the application.


# Development

**IVyTools** is preliminary or provisional and is subject to revision. It is 
being provided to meet the need for timely best science. The software has 
not received final approval by the U.S. Geological Survey (USGS). No 
warranty, expressed or implied, is made by the USGS or the U.S. Government 
as to the functionality of the software and related material nor shall the 
fact of release constitute any such warranty. The software is provided on 
the condition that neither the USGS nor the U.S. Government shall be held 
liable for any damages resulting from the authorized or unauthorized use of 
the software.

## Running IVyTools from source
Instructions on setting up the IVyTools environment and running from source 
can be found [here](docs/source/setup.md).


## Additional Publication Details

This table contains extra information about the publication that cannot be 
found elsewhere on the page.

| **Publication Type** |  Python module   |
|----------------------|:----------------:|
| **DOI**              | 10.5066/P1KMVCNY |
| **Year published**   |       2025       |
| **Year of version**  |       2025       |
| **Version**          |     1.0.0.0      |
| **IPDS**             |    IP-173457     |


