import os
import unittest

from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

unittest.TestLoader.sortTestMethodsUsing = None

if __name__ == "__main__":
    assignment_tag = os.environ.get("assignment_tag", None)
    tests_dir = (
        os.path.join("tests", assignment_tag) if assignment_tag is not None else "tests"
    )
    suite = unittest.defaultTestLoader.discover(tests_dir)
    with open("/autograder/results/results.json", "w") as f:
        JSONTestRunner(stream=f).run(suite)
