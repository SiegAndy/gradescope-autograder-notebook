from gradescope_utils.autograder_utils.decorators import number, visibility, weight
from tests import (
    TestJupyterNotebook,
    tokenize_space,
    tokenize_fancy,
    tokenize_4grams,
    tokenization,
    stopping,
    stemming,
)


class TestNotebookCompilable(TestJupyterNotebook):
    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            jupyter_notebook_file_path="446_p1.ipynb",
            data_file_path="links-ireland.srt.gz",
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
        try:
            submission_method = self.client.ref("tokenize_space")
            print(submission_method("aaa aaa"))
            for curr_sentence in self.sentences:
                golden_result = tokenize_space(curr_sentence)
                self.client.inject(r"%reset -f in out", pop=True)
                print(submission_method(curr_sentence), curr_sentence)
                target_result = self.method_wrapper(submission_method, curr_sentence)
                self.assertEqual(
                    len(golden_result),
                    len(target_result),
                    "Output length does not match.\n"
                    + f"Expect {len(golden_result)} tokens but {len(target_result)} received!"
                    + f"\n\t Expected results: [{', '.join(golden_result)}]",
                )
                self.assertEqual(
                    sorted(golden_result),
                    sorted(target_result),
                    "Output does not match.\n"
                    + f"Expect results: [{', '.join(golden_result)}]\n"
                    + f"but received: [{', '.join(target_result)}]",
                )
        except Exception as e:
            raise
            self.assertTrue(False, f"Code does not compile:\n{e.args}")

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
