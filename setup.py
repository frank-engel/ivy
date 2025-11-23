from setuptools import setup
# read the contents of your README file
from pathlib import Path

# Build instructions
# Run: package_app.py

# Freeze requirements instructions
# pip uninstall image_velocimetry_tools
# python ./setup.py sdist
# pip install image_velocimetry_tools
# pip freeze > requirements.txt
# ( check that requirements.txt was utf-8 encoded, if not resave using Notepad++ or similar)

# Code summary with pygount (not in requirements,
# but needs to be installed in your environment to run
# >> pygount.exe --format=summary .\image_velocimetry_tools\


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the dependencies from install_requires.txt
# Note: requirements.txt contains exact frozen versions for development
# install_requires.txt contains relaxed constraints for distribution
with open('install_requires.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='image_velocimetry_tools',
    version='1.0.0.2',
    packages=[
        'image_velocimetry_tools',
        'image_velocimetry_tools.gui',
        'image_velocimetry_tools.batch',
        'image_velocimetry_tools.batch.models',
        'image_velocimetry_tools.services',
        'image_velocimetry_tools.api',
    ],
    py_modules=['image_velocimetry_tools'],
    python_requires='>=3.11',
    url='https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy',
    license='GNU General Public License v3.0',
    license_files=["LICEN[CS]E*", "AUTHORS.md"],
    author='Frank L. Engel and Travis Knight',
    author_email='fengel@usgs.gov',
    description='A package and gui application for computing water velocity using videos.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    entry_points={
        'console_scripts': [],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)
