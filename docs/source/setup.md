# Setting up the development environment

The instructions listed below guide the user through the process of running 
IVyTools using the source code. The information below will walk the user 
through cloning the repository, creating a development environment, and 
installing the IVyTools package.

**TL;DR:** Presuming the `ivy` environment and dependencies are all 
correctly installed, start IVyTools from a PowerShell terminal like this:
```powershell
$env:IVY_ENV = "development"
python .\app.py
```

## Clone the IVyTools repository
The best way to obtain these packages is to clone the repository.

In Git Bash, navigate to the directory where you want to store the
repositories. When you clone a repository, a new directory with the name of
the project will be created, and the directory will contain the repository. To
clone the IVyTools repository, type the following command.

```
$ git clone git@code.usgs.gov:hydrologic-remote-sensing-branch/ivy.git
```

## Change directories
Before you continue, change directories into the IVyTools directory by 
typing the
following command.

```
$ cd ivy
```

## Set up a virtual environment
When creating a virtual environment, you will create an isolated 
installation of Python and install the supporting packages with the correct 
versions. The environment will be created within the IVyTools directory, 
and it will stand apart from other Python environments on your machine.

## Create a virtual environment
Create an environment using the venv module.

```
$ python -m venv c:\path\to\myenv
```

See the [Virtual Environments and Packages](https://docs.python.org/3.7/tutorial/venv.html) and
[venv](https://docs.python.org/3.7/library/venv.html) documentation for more
information.

## Activate the virtual environment
Use the `activate` command to activate the virtual environment. The name of the
environment will appear in parentheses in the shell prompt. To activate the
environment in the Git Bash terminal, type the following command.

```
$ . env/Scripts/activate
(env) $
```

To activate the virtual environment in the Windows command prompt (outside of
this tutorial, for instance), type the following command

```
>env\Scripts\activate
```

You must activate the virtual environment when you work within the `ivy`
environment in the future. 

## Install dependencies
In order for the Python interpreter in the IVyTools environment to have 
"global"
access to the package dependencies, you'll have to install the packages within the
IVyTools environment.

This command assumes you're working in the IVyTools directory and have cloned the
repositories in the parent directory.

```
(env) $ pip install -e .
```

pip is the standard Python package management system. The `-e` option tells pip
to install the package in "editable" mode (see
[Editable installs](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
for more info). `../ivy` tells pip to look for a setup.py file in
the directory named `ivy` which is one level above the current
working directory.

### Install the required Python packages
In the IVyTools repository, there is a file named `requirements.txt` that
contains a list of Python packages that are required by IVyTools. The
requirements file is used with pip to install the required packages in the new
environment.

Type the following command to install the required packages in the new
environment.

```
(env) $ pip install -r requirements.txt
```

### Install required 3rd party binaries
In addition to the Python packages, IVyTools relies on a few external tools 
that must be installed and properly configured.

#### 1. Install ffmpeg and ffprobe
IVyTools requires the `ffmpeg` and `ffprobe` executables for video processing. 
These are not installed via pip and must be manually downloaded.

##### Steps:
1. Go to the [FFmpeg download page](https://ffmpeg.org/download.html), or use 
   a precompiled Windows build from [gyan.dev](gyan.dev).
2. Download a static build and extract it. The "essentials" build contains the 
   needed libraries.
3. Locate `ffmpeg.exe` and `ffprobe.exe` in the extracted `bin` directory.
4. Copy both ffmpeg.exe and ffprobe.exe into the following directory relative 
   to the IVyTools root:
   ```bash
     ../ivy/bin/
   ```
   Create the `bin` directory if it doesn't exist.

You can verify this step was successful by checking:

```bash
../ivy/bin/ffmpeg.exe
../ivy/bin/ffprobe.exe
```

Note: when IVyTools is installed using the MSI Self Installer, it will 
place `ffmpeg` and `ffprobe` in the `%ProgramData%/IvyTools` directory (e.g.
, `C:/ProgramData/IvyTools`) and will add two system variable, 
`FFMPEG-IVyTools` and `FFPROBE-IVyTools` to the path.

#### 2. Install LAVFilters
LAVFilters is used to decode a wide variety of video formats through the 
Windows DirectShow interface.

##### Steps:
1. Download the latest LAVFilters installer from the official website: 
   https://github.com/Nevcairiel/LAVFilters/releases
2. Run the installer and follow the installation steps.
3. Make sure the option to install LAV Video and Audio decoders is selected.

###### Verifying the installation:
* After installation, open the Start menu and search for "LAV Video 
  Configuration" or "LAV Audio Configuration".
  * If the configuration utilities open, LAVFilters is correctly installed.
* You can also check for installed files at:
   ```powershell
   C:\Program Files (x86)\LAV Filters\
   ```
* During video playback in compatible media players, you should see LAV icons 
  appear in the Windows system tray.

## Building the documentation
Once IVyTools and the dependencies have been installed the documentation 
files can be updated by running Sphinx. To build the documentation open a 
terminal, activate the IVyTools environment, then navigate to the docs 
directory within the  IVyTools folder  structure (`../ivy/docs`). In 
the terminal type:

```
./make html
```

## Starting IvyTools from source
Once the environment, third-party dependencies, and documentation steps 
above are complete, the IVyTools application can be started in 
`development` mode from a terminal like this:

### In PowerShell:
```powershell
$env:IVY_ENV = "development"
python .\app.py
```

### In Command Prompt (cmd.exe):
```cmd
set IVY_ENV=development
python app.py
```

### In unix-style shells:
```bash
IVY_ENV=development python .\app.py
```