import ast
import asyncio
import io
import os
import sys
import unittest
from typing import Any, Callable, List

from gradescope_utils.autograder_utils.files import SUBMISSION_BASE
from testbook import testbook
from testbook.client import TestbookNotebookClient
from tqdm import tqdm

from default_import import import_checker_stmt
from solution import (
    load_file,
    tokenize_space,
    tokenize_fancy,
    tokenize_4grams,
    stemming,
    stopping,
    tokenization,
    heaps,
    statistics,
)

DATAPATH = "./data"

if sys.platform.startswith("win"):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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
        jupyter_notebook_file_path: str,
        data_file_path: str,
        allowed_imports: List[str] = None,
    ):
        cls.original_stdout = sys.stdout
        cls.suppress_text = io.StringIO()
        cls.allowed_imports = allowed_imports
        cls.jupyter_notebook_file_path = os.path.join(
            SUBMISSION_BASE, jupyter_notebook_file_path
        )
        cls.sentences = load_file(
            data_file_path,
            gz_zip=True if data_file_path.endswith(".gz") else False,
        )
        cls.stopwords = load_file(
            "stopwords.txt",
            gz_zip=False,
        )
        cls.notebook = testbook(cls.jupyter_notebook_file_path, execute=False)
        cls.is_compilable = None
        cls.imported_disallowed_pkgs = None
        cls.err = None
        cls.err_has_been_reported = None

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

        if self.err_has_been_reported is None:
            return

        # only report detailed error once.
        if self.err_has_been_reported:
            msg = "See error message above."
            return self.assertTrue(
                False,
                msg=msg,
            )

        self.err_has_been_reported = True
        if not self.is_compilable:
            self.exit_kernel()
            msg = f"The notebook is not compilable! Detail: \n{self.err}"
            self.assertTrue(
                self.is_compilable,
                msg=msg,
            )

        if len(self.imported_disallowed_pkgs) > 0:
            self.assertIn(
                self.imported_disallowed_pkgs,
                self.allowed_imports,
                f"Import(s) not allowed: {', '.join(f'<{name}>' for name in self.imported_disallowed_pkgs)}.",
            )

    def clear_notebook_output(self, curr_line_count: int, line_limits: int = 50):
        if curr_line_count % line_limits == 0:
            self.client.inject(r"%reset -f in out", pop=True)
