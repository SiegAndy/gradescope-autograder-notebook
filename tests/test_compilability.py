from gradescope_utils.autograder_utils.decorators import number, visibility, weight
from tests import TestJupyterNotebook


class TestNotebookCompilable(TestJupyterNotebook):
    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            jupyter_notebook_file_path="446_p1_solution.ipynb",
            data_file_path="my-links-ireland.srt.gz",
            allowed_imports=["re"],
            # allowed_imports=["gzip", "pathlib"],
        )
        return

    @weight(0)
    @visibility("visible")
    @number("1.1")
    def test_ipynb_compilable_and_packages(self):
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("1.2")
    def test_sample_tokenization_token_only(self):
        self.checker()

    # @weight(0)
    # @visibility("visible")
    # @number("1.3")
    # def test_sample_stopping(self):
    #     self.checker()

    # @weight(0)
    # @visibility("visible")
    # @number("1.4")
    # def test_sample_stemming(self):
    #     self.checker()

    # @weight(0)
    # @visibility("visible")
    # @number("1.5")
    # def test_sample_tokenization(self):
    #     self.checker()

    # @weight(0)
    # @visibility("visible")
    # @number("1.6")
    # def test_sample_statistics(self):
    #     self.checker()
