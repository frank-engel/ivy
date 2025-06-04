# Running Tests with Coverage
Note that when these tests are run, it uses the currently installed 
version of the module. This means that it is good practice to reinstall 
the module with `pip install` prior to running the coverage.

In order to run all unittests for IVy with Coverage reporting in PyCharm, 
ensure that `coverage` is installed, and type the 
following command _from the `image_velocimetry_framework/tests` directory_:

```bash
cd ./tests
coverage run --source=./ -m unittest discover -s . ; coverage html
```