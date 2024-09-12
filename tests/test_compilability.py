import io
import json
import os
import shutil
import sys
import types
import unittest
from typing import Any, Callable

from gradescope_utils.autograder_utils.decorators import number, weight, visibility
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
from testbook import testbook
from tests import SUBMISSION_BASE, TestJupyterNotebook


class TestNotebookCompilable(TestJupyterNotebook):

    def setUp(self):
        super().setUp(
            jupyter_notebook_file_path="test.ipynb",
            data_file_path="my-links-ireland.srt.gz",
            allowed_imports=["gzip"],
        )
        self.compilablity()
        return

    @weight(0)
    @visibility("visible")
    @number("1.1")
    def test_ipynb_compilable(self):
        self.assert_compilable()

    @weight(0)
    @visibility("visible")
    @number("1.2")
    def test_import(self):
        """Check imported packages"""
        self.import_checker()

    @weight(0)
    @visibility("visible")
    @number("1.3")
    def test_sample_tokenization_token_only(self):
        self.assert_compilable()
