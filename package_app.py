import os
import re
import sys
from datetime import datetime
import json
import click
import shutil
import subprocess
import unittest
import pyinstaller_versionfile
import PyInstaller.__main__

from image_velocimetry_tools import __version__, __credits__


def create_dist_directory(dist_path):
    """Create the distribution directory if it doesn't exist."""
    if not os.path.exists(dist_path):
        os.mkdir(dist_path)


def confirm_and_clean_existing_version(version_dir, version):
    """Check if the package version already exists and clean it if confirmed."""
    if os.path.exists(version_dir):
        if click.confirm(
            f"Version {version} already exists. Overwrite?", default=True, abort=True
        ):
            try:
                shutil.rmtree(version_dir)
            except BaseException:
                print("Something went wrong, exiting build.")
                sys.exit()


def update_file_version(file_path, version_pattern, version):
    """Update the version string in a specified file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        updated_content = re.sub(version_pattern, f"version='{version}'", content)
        with open(file_path, 'w') as file:
            file.write(updated_content)


def build_sphinx_docs(docs_dir):
    """Build Sphinx documentation."""
    print("Building Sphinx documentation...")
    make_cmd = os.path.join(docs_dir, "make.bat") if os.name == 'nt' else "make"
    try:
        subprocess.run([make_cmd, "html"], cwd=docs_dir, check=True)
        print("Sphinx documentation built successfully.")
    except FileNotFoundError:
        print(f"Error: {make_cmd} not found. Ensure Sphinx is installed and the Makefile exists.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while building Sphinx documentation: {e}")


def create_version_file(version, credits, file_description, internal_name, exe_name, product_name):
    """Create a version file for the application."""
    pyinstaller_versionfile.create_versionfile(
        output_file="file_version_info.txt",
        version=version,
        company_name=credits,
        file_description=file_description,
        internal_name=internal_name,
        legal_copyright="CC0 1.0",
        original_filename=f"{exe_name}",
        product_name=product_name,
        translations=[1033, 1200],
    )


def update_nested_urls(data, keys, version_fragment):
    """Recursively update URLs in nested dictionaries."""
    for key in keys:
        if key in data:
            if isinstance(data[key], list):  # Handle lists of dictionaries
                for item in data[key]:
                    if isinstance(item, dict):
                        update_nested_urls(item, keys, version_fragment)
                    elif isinstance(item, str) and '/0.' in item:
                        data[key] = re.sub(r'/0\.\d+\.\d+/', version_fragment, item)
            elif isinstance(data[key], dict):  # Nested dictionary
                update_nested_urls(data[key], keys, version_fragment)
            elif isinstance(data[key], str) and '/0.' in data[key]:
                data[key] = re.sub(r'/0\.\d+\.\d+/', version_fragment, data[key])


def update_code_json_versioning(code_json_path, version):
    """Update the version and URLs in code.json to reference the immutable tag."""
    if not os.path.exists(code_json_path):
        print(f"{code_json_path} not found.")
        return

    with open(code_json_path, 'r') as file:
        code_data = json.load(file)

    def revise_url(url):
        # Replace any -/raw/main/ or -/archive/main/ with the versioned tag
        url = re.sub(r'(-/raw|-/archive)/main/', rf'\1/{version}/', url)
        return url

    url_keys = ['URL', 'homepageURL', 'downloadURL', 'disclaimerURL', 'repositoryURL']

    def update_urls(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in url_keys and isinstance(value, str):
                    obj[key] = revise_url(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    update_urls(value)
        elif isinstance(obj, list):
            for item in obj:
                update_urls(item)

    update_urls(code_data)

    # Set the version field and update metadata date
    code_data[0]['version'] = version
    today = datetime.now().strftime('%Y-%m-%d')
    code_data[0]['date']['metadataLastUpdated'] = today

    with open(code_json_path, 'w') as file:
        json.dump(code_data, file, indent=2)

    print("code.json updated successfully.")


def run_tests_in_directory(test_dir):
    # Change the current working directory to the test directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)

    try:
        # Add the parent directory of the test directory to sys.path
        sys.path.insert(0, os.path.abspath(os.path.join(test_dir, os.pardir)))

        # Discover all test cases in the specified directory
        loader = unittest.TestLoader()
        suite = loader.discover('.')

        # Create a test runner
        runner = unittest.TextTestRunner()

        # Run the tests
        result = runner.run(suite)

        # Check for failures or errors and print a warning if found
        if not result.wasSuccessful():
            print(f"WARNING: Some tests did not pass. Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)
        return result


def build_application(
    version,
    credits,
    product_name,
    internal_name,
    exe_name,
    file_description,
    package_prefix
):
    """Main function to build the application."""
    print("Starting build process...")

    ivy_package = f"{package_prefix}_v{version.replace('.', '')}"
    dist_dir = os.path.join(os.getcwd(), "dist")
    app_dir = os.path.join(dist_dir, ivy_package)
    build_dir = os.path.join(os.getcwd(), "build")

    # Step 0: Run tests and ensure passing
    # test_results = run_tests_in_directory("./image_velocimetry_tools/tests")
    #
    # if test_results is None:
    #     return

    # Step 1: Check and clean existing version
    confirm_and_clean_existing_version(app_dir, version)

    # Step 2: Ensure distribution directory exists
    create_dist_directory(dist_dir)
    os.mkdir(app_dir)

    # Step 3: Update version in relevant files
    update_file_version(os.path.join(os.getcwd(), "setup.py"), r"version\s*=\s*['\"]([^'\"]*)['\"]", version)
    update_file_version(os.path.join(os.getcwd(), "docs/conf.py"), r"version\s*=\s*['\"]([^'\"]*)['\"]", version)

    # Step 4: Build Sphinx documentation
    build_sphinx_docs(os.path.join(os.getcwd(), "docs"))

    # Step 5: Create version file
    create_version_file(version, credits, file_description, internal_name, exe_name, product_name)

    # Step 6: Update code.json
    update_code_json_versioning(os.path.join(os.getcwd(), "code.json"), version)

    # Step 7: Run PyInstaller
    print("Running PyInstaller...")
    PyInstaller.__main__.run([
        "app.spec",
        "--distpath", app_dir,
        "--workpath", build_dir,
        "--noconfirm"
    ])

    # Step 8: Verify and clean up
    build_exe = os.path.join(app_dir, exe_name)
    if os.path.isfile(build_exe):
        os.remove(build_exe)

    # Move contents of app_dir/app_name to app_dir
    built_subdir = os.path.join(app_dir, internal_name)
    if os.path.isdir(built_subdir):
        for item in os.listdir(built_subdir):
            src = os.path.join(built_subdir, item)
            dst = os.path.join(app_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(built_subdir)

    print(
        f"Please sign the executable in '{app_dir}' before zipping. Packaging complete.")


if __name__ == "__main__":
    # Example: Pass all parameters dynamically or from environment
    build_application(
        version=__version__,
        credits=__credits__,
        product_name="Image Velocimetry Tools",
        internal_name="IVyTools",
        exe_name="IvyTools.exe",
        file_description="Image Velocimetry Tools",
        package_prefix="IVyTools"
    )