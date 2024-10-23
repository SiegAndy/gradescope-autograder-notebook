import ast
import asyncio
import io
import os
import sys
import unittest
from typing import Any, Callable, List

from default_import import import_checker_stmt
from gradescope_utils.autograder_utils.files import SUBMISSION_BASE
from testbook import testbook
from testbook.client import TestbookNotebookClient
from tqdm import tqdm

from solution import (
    heaps,
    load_file,
    statistics,
    stemming_porter,
    stemming_s,
    stopping,
    tokenization,
    tokenize_4grams,
    tokenize_fancy,
    tokenize_space,
)
from solution import version as solution_version

DATAPATH = "./data"

if sys.platform.startswith("win"):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

in_gradescope = True if os.environ.get("in_gradescope", False) else False

tqdm = tqdm if not in_gradescope else lambda x, **_: x


# def exception_catcher(self: unittest.TestCase, func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except AssertionError:
#             raise
#         except Exception as e:
#             self.assertTrue(False, f"Code does not compile:\n{e}")

#     return wrapper


def pick_up_submission_notebook(submission_folder: str = SUBMISSION_BASE):
    # List all files in the given folder
    files = os.listdir(submission_folder)

    # Filter files to include only .ipynb files
    ipynb_files = [f for f in files if f.endswith(".ipynb")]

    # Return the first .ipynb file if it exists, otherwise None
    return ipynb_files[0] if ipynb_files else None


class TestJupyterNotebook(unittest.TestCase):
    jupyter_notebook_file_path: str  # path to jupyter notebook
    notebook: testbook  # notebook class, use context manager to retrieve client
    sentences: List[str]  # loaded sentence from dataset
    stopwords: List[str]  # loaded stopwords from dataset
    is_compilable: bool  # is notebook compilable
    is_imports_allowed: bool  # is imports in notebook are all allowed
    err: Exception  # exception happened when try to run cell(s)
    allowed_imports: List[str]
    client: TestbookNotebookClient

    @classmethod
    def setUpClass(
        cls,
    ):
        cls.original_stdout = sys.stdout
        cls.suppress_text = io.StringIO()

        cls.is_compilable = None
        cls.imported_disallowed_pkgs = None
        cls.err = None
        cls.err_has_been_reported = None

        jupyter_notebook_file_path = pick_up_submission_notebook(SUBMISSION_BASE)

        if jupyter_notebook_file_path is None:
            cls.is_compilable = False
            cls.err_has_been_reported = False
            cls.err = "Fail to find an valid notebook file in the submission! Please upload a valid jupyter notebook file!"
        cls.jupyter_notebook_file_path = os.path.join(
            SUBMISSION_BASE, jupyter_notebook_file_path
        )
        cls.notebook = testbook(cls.jupyter_notebook_file_path, execute=False)

        cls.setUp_kernel(cls)
        cls.compilablity()

    @classmethod
    def tearDownClass(cls):
        cls.exit_kernel(self=cls)

    @classmethod
    def compilablity(cls) -> None:
        # no need to check again
        if cls.is_compilable is not None:
            return

        cls.suppress_print(cls, True)
        try:
            # with self.notebook as notebook_kernel:
            # print(json.dumps(notebook_kernel.cells, indent=4))
            cls.client.execute()
            student_version = cls.client.ref("version")
            if student_version != solution_version:
                raise ValueError(
                    f"Version Mismatched. Expects notebook version: '{student_version}', but submission has version: '{solution_version}'!"
                )
            cls.suppress_print(cls, False)
            cls.is_compilable = True
            cls.err = None
        except Exception as e:
            # close the notebook kernel as exception occure
            cls.exit_kernel(cls)
            cls.suppress_print(cls, False)
            cls.is_compilable = False
            cls.err = e
            cls.err_has_been_reported = False

    # def __getattribute__(self, name):
    #     attr = super().__getattribute__(name)
    #     # Apply the decorator to methods that start with 'test'
    #     if name.startswith("test") and callable(attr):
    #         return exception_catcher(self, attr)
    #     return attr

    def setUp_kernel(self) -> TestbookNotebookClient:
        with self.notebook.client.setup_kernel(cleanup_kc=False):
            self.notebook._prepare()
            self.client = self.notebook.client
            return self.client

    def exit_kernel(self) -> None:
        if (
            hasattr(self, "client")
            and hasattr(self.client, "km")
            and self.client.km is not None
            and asyncio.run(self.client.km.is_alive())
        ):
            self.client._cleanup_kernel()

    def suppress_print(self, suppress_print: bool) -> None:
        if suppress_print:
            sys.stdout = self.suppress_text
        else:
            sys.stdout = self.original_stdout

    def method_wrapper(
        self, method: Callable, *inputs, suppress_print: bool = True, **kargs
    ) -> Any:
        if suppress_print:
            self.suppress_print(True)
        result = method(*inputs, **kargs)
        if suppress_print:
            self.suppress_print(False)
        return result

    def import_checker(self):
        """Check imported packages"""
        if not self.is_compilable:
            return

        self.imported_disallowed_pkgs = None
        if self.allowed_imports is None:
            return

        # direct inject code into notebook for checking the modules, a workaround
        # for directly accessing globals().keys(), as notebook.ref() need to
        # inject code into notebook which modify the globals() dynamically.
        node = self.client.inject(
            code=import_checker_stmt(self.allowed_imports), pop=True
        )
        self.imported_disallowed_pkgs = ast.literal_eval(
            node.execute_result[0]["text/plain"]
        )
        # print(result)

    def checker(self) -> None:

        if self.__class__.err_has_been_reported is None:
            return

        # only report detailed error once.
        if self.__class__.err_has_been_reported:
            msg = "See error message above."
            return self.assertTrue(
                False,
                msg=msg,
            )

        self.__class__.err_has_been_reported = True
        if not self.__class__.is_compilable:
            self.__class__.exit_kernel(self=self)
            msg = f"The notebook is not compilable! Detail: \n{self.__class__.err}"
            self.assertTrue(
                self.__class__.is_compilable,
                msg=msg,
            )

        if len(self.__class__.imported_disallowed_pkgs) > 0:
            self.assertIn(
                self.__class__.imported_disallowed_pkgs,
                self.__class__.allowed_imports,
                f"Import(s) not allowed: {', '.join(f'<{name}>' for name in self.__class__.imported_disallowed_pkgs)}.",
            )

    def clear_notebook_output(self, curr_line_count: int, line_limits: int = 50):
        if curr_line_count % line_limits == 0:
            self.client.inject(r"%reset -f in out", pop=True)
