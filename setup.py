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

# Read the dependencies from requirements.txt
with open('requirements.txt', encoding='utf-16') as f:
    requirements = f.read().splitlines()

setup(
    name='image_velocimetry_tools',
    version='1.0.0.0',
    packages=['image_velocimetry_tools', 'image_velocimetry_tools.gui'],
    py_modules=['image_velocimetry_tools'],
    url='https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy',
    license='GNU General Public License v3.0',
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
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)
