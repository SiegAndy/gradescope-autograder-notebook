import asyncio
import io
import json
import os
import sys
import types
import unittest
from typing import Any, Callable, Dict, List
from testbook import testbook

from gradescope_utils.autograder_utils.files import SUBMISSION_BASE

# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class SuppressClass(io.StringIO):
    def __init__(
        self,
        *args,
    ):
        io.StringIO.__init__(self, *args)

    def getvalue(self):
        return ""


class TestJupyterNotebook(unittest.TestCase):
    jupyter_notebook_file_path: str  # path to jupyter notebook
    data_file_path: str  # path to dataset
    notebook: testbook  # notebook class, use context manager to retrieve client
    is_compilable: bool  # default to True
    err: Exception  # exception happened when try to run cell(s)
    allowed_imports: List[str]

    def setUp(
        self,
        jupyter_notebook_file_path: str,
        data_file_path: str,
        allowed_imports: List[str] = None,
    ):
        self.original_stdout = sys.stdout
        self.suppress_text = io.StringIO()
        self.allowed_imports = allowed_imports
        self.jupyter_notebook_file_path = os.path.join(
            SUBMISSION_BASE, jupyter_notebook_file_path
        )
        self.data_file_path = os.path.join(
            SUBMISSION_BASE, data_file_path
        )  # TODO: load dataset
        self.notebook = testbook(self.jupyter_notebook_file_path, execute=False)
        self.is_compilable = True
        self.err = None
        self.err_has_been_reported = False

    def suppress_print(self, suppress_print: bool) -> None:
        if suppress_print:
            sys.stdout = self.suppress_text
        else:
            sys.stdout = self.original_stdout

    def method_wrapper(
        self, method: Callable, *inputs, suppress_print: bool = True
    ) -> Any:
        if suppress_print:
            self.suppress_print(True)
        result = method(*inputs)
        if suppress_print:
            self.suppress_print(False)
        return result

    def compilablity(self) -> None:
        self.suppress_print(True)
        try:
            with self.notebook as notebook_kernel:
                print(json.dumps(notebook_kernel.cells, indent=4))
                notebook_kernel.execute()
                self.suppress_print(False)
        except Exception as e:
            self.suppress_print(False)
            self.is_compilable = False
            self.err = e
            self.err_has_been_reported = False

    def assert_compilable(self) -> None:

        # only report detailed error once.
        msg = "See error message above."
        if not self.err_has_been_reported:
            msg = f"The notebook is not compilable! Detail: \n{self.err}"
            self.err_has_been_reported = True

        self.assertTrue(
            self.is_compilable,
            msg=msg,
        )

    def import_checker(self):
        """Check imported packages"""
        self.assert_compilable()

        if self.allowed_imports is None:
            return True

        for name, val in list(globals().get("RSA").__dict__.items()):
            if name.startswith("__"):
                continue
            if (
                isinstance(val, types.BuiltinMethodType)
                or isinstance(val, types.BuiltinFunctionType)
                or isinstance(val, types.FunctionType)
            ):
                continue
            if "typing." in str(val):
                continue
            return self.assertIn(
                name,
                self.allowed_imports,
                f"Import not allowed: <{name}>",
            )
