import io
import os
import shutil
import sys
import unittest
from typing import Any, Callable

from gradescope_utils.autograder_utils.decorators import number, weight
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
from testbook import testbook
from tests import SUBMISSION_BASE


class SuppressClass(io.StringIO):
    def __init__(
        self,
        *args,
    ):
        io.StringIO.__init__(self, *args)

    def getvalue(self):
        return ""

class TestNotebook(unittest.TestCase):

    def setUp(self):
        self.original_stdout = sys.stdout
        self.suppress_text = io.StringIO()
        self.file_path = os.path.join(SUBMISSION_BASE, "test.ipynb")

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

    @weight(1)
    @number("1.1")
    def test_task_1_1(self):
        # self.suppress_print(True)
        try:
            print("pass1")
            with testbook(self.file_path, execute=True) as tb:
                tb.execute_cell
                print("pass2")
                task_1_1 = tb.ref("func")
                print("pass3")
                print("*********** Hello ***************")
                print("pass4")
                assert task_1_1(1,2) == 3
                print("pass5")
                self.suppress_print(False)
        except Exception as e:
            self.suppress_print(False)
            print(e)            
    
    # def test_eval_add(self):
    #     """Evaluate 1 + 1"""
    #     val = self.calc.eval("1 + 1")
    #     self.assertEqual(val, 2)

    # @weight(1)
    # @number("1.2")
    # def test_eval_sub(self):
    #     """Evaluate 2 - 1"""
    #     val = self.calc.eval("2 - 1")
    #     self.assertEqual(val, 1)

    # @weight(1)
    # @number("1.3")
    # def test_eval_mul(self):
    #     """Evaluate 4 * 8"""
    #     val = self.calc.eval("4 * 8")
    #     self.assertEqual(val, 32)

    # @weight(1)
    # @number("1.4")
    # def test_eval_div(self):
    #     """Evaluate 8/4"""
    #     val = self.calc.eval("8 / 4")
    #     self.assertEqual(val, 2)

    # @weight(2)
    # @number("1.5")
    # def test_eval_whitespace(self):
    #     """Evaluate 1+1 (no whitespace)"""
    #     val = self.calc.eval("1+1")
    #     self.assertEqual(val, 2)