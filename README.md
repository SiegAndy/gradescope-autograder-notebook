# Gradescope-Autograder-Notebook

This repo is designed to autograde the jupyter notebook submission on gradescope. It contains a custom unittest class which simulate the process of open the notebook, run all the cells, and reflect the variables of the notebook to grade the submission.


## How to run

In local environment, after installing necessary packages, run `python3 text_tests.py`.

In gradescope environment, the gradescope will handle the process of turning zipped files into docker image and host it in a virtual machine and run `python3 run_tests.py`.

## What need to upload to gradescope
Here are a list of required files:
* `make_autograder.sh`: the script zip all following necessary files into one zip file which would be uploaded to gradescope.
* `setup.sh`: the setup script of gradescope docker image.
* `run_autograder.sh`: the script (possibly the entrypoint) of gradescope docker image.
* `run_tests.py`: the python script which loads all the tests from `tests/` folder and run it on gradescope.
* `text_tests.py`: the python script which loads all the tests from `tests/` folder and run it locally (for debugging).
* `requirements.txt` (optional): the file contains required python package. Note, you can done the same thing in the `setup.sh`.
* `tests/test_*.py`: the python scripts which contain a list of test cases.
* `data/*`: the folder which contains a list of required data such as dataset and stopwords.

The submission will be placed at `/autograder/submission` on the gradescope. To test submission locally (debugging), you can modify the `SUBMISSION_BASE` variable from `gradescope_util` package.

**Modify the files in `make_autograder.sh`. Then, executing this script will zip all necessary files into `autograder.zip`.**

## Notebook Autograder Class&Utility

In `tests/__init__.py`, a few utilities are provided. 
* `exception_catcher`: the decorater function that catch every exception except for the AssertionError (which is used by gradescope to determine whether the test is fail or not.)
* `TestJupyterNotebook`: the custom unittest class which loads the first jupyter notebook file in the submission folder and perform reflection on the variables. The class 
  * provides a method wrapper `method_wrapper()` which allows the autograder `suppress_print()` the text output from the given method. For example, it can suppress the `print` output from the submission notebook.
  * handles the connection to jupyter notebook's kernel internally, without the need of using context manager from `notebook` package.
  * implements multiple error checking and notebook output removal utilities.

## COMPSCI 446 PA1
In `tests/pa1.py`, a unittest class `TestPA1` designed specifically for the programming assignment 1 of COMPSCI446: Search Engine (Fall 2024).

All the unittest class in `test_compilability.py` and `test_tokenization.py` are inherited from `TestPA1` class.

The class `TestPA1` provides additional utilities:
* check and store stopwords and dataset in the class variable.
* order the test cases in a way such that the previous test case result is stored and used in the next test case (a bit towards integral-testing).
* `prerequisite_tester()` and `no_prerequisite_tester()` methods are coded to reduce the code complexity as most test cases share the similar structure:
  * check whether prerequisite test case(s) are success;
  * try to reflect the designated method from jupyter notebook kernel;
  * utilize stored class variable and call the solution/relfected method with the stored class variable(s);
  * assert whether relfected method produce the same result as the solution method;
  * if tests are passed, store this round result in class variable for later test case usage.