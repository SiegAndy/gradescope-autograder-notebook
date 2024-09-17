from typing import Any, Callable, List, Tuple
from gradescope_utils.autograder_utils.decorators import number, visibility, weight
from tests import (
    TestJupyterNotebook,
    tokenize_space,
    tokenize_fancy,
    tokenize_4grams,
    tokenization,
    stopping,
    stemming,
    heaps,
    statistics,
)
from tqdm import tqdm


class TestNotebookCompilable(TestJupyterNotebook):
    tokenized_fancy_solution: List[str]
    tokenized_fancy_student: List[str]
    tokenized_fancy_with_stopping_solution: List[str]
    tokenized_fancy_with_stopping_student: List[str]
    tokenization_solution: List[Tuple[str, List[str]]]
    tokenization_student: List[Tuple[str, List[str]]]

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            jupyter_notebook_file_path="446_p1.ipynb",
            data_file_path="test2",
            # data_file_path="P1-train.gz",
            allowed_imports=["re"],
            # allowed_imports=["gzip", "pathlib"],
        )
        return

    @weight(0)
    @visibility("visible")
    @number("1.1")
    def test_01_ipynb_compilable_and_packages(self):
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("1.2")
    def test_02_sample_tokenization_space(self):
        self.tokenization_tester(
            function_name="tokenize_space",
            solution_function=tokenize_space,
            tqdm_desc="test_tokenize_space",
        )

    @weight(0)
    @visibility("visible")
    @number("1.3")
    def test_03_sample_tokenization_fancy(self):
        self.tokenization_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tqdm_desc="test_tokenize_fancy",
        )

    @weight(0)
    @visibility("visible")
    @number("1.4")
    def test_04_sample_tokenization_4grams(self):
        self.tokenization_tester(
            function_name="tokenize_4grams",
            solution_function=tokenize_4grams,
            tqdm_desc="test_tokenize_4grams",
        )

    @weight(0)
    @visibility("visible")
    @number("1.5")
    def test_05_sample_tokenization_fancy_with_stopping(self):
        self.checker()
        try:
            # get target function reflection
            stopping_method = self.client.ref("stopping")

            self.__class__.tokenized_fancy_with_stopping_solution = []
            self.__class__.tokenized_fancy_with_stopping_student = []

            self.assertEqual(
                len(self.sentences),
                len(self.tokenized_fancy_student),
                "Need to first pass the test: tokenization_fancy!",
            )

            for idx, (curr_sentence, solution_tokens, student_tokens) in tqdm(
                enumerate(
                    zip(
                        self.sentences,
                        self.tokenized_fancy_solution,
                        self.tokenized_fancy_student,
                    )
                ),
                desc="test_stopping",
            ):
                # tokenize the sentence using solution code
                golden_result = stopping(solution_tokens, self.stopwords)

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                stopping_results = self.method_wrapper(
                    stopping_method, student_tokens, self.stopwords
                )

                self.assertion_set(
                    curr_sentence=curr_sentence,
                    golden_results=golden_result,
                    tokenized_results=stopping_results,
                )

                self.__class__.tokenized_fancy_with_stopping_solution.append(
                    golden_result
                )
                self.__class__.tokenized_fancy_with_stopping_student.append(
                    stopping_results
                )
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    @weight(0)
    @visibility("visible")
    @number("1.6")
    def test_06_sample_tokenization_fancy_with_stopping_and_stemming(self):
        self.checker()
        try:
            # get target function reflection
            stemming_method = self.client.ref("stemming")

            self.assertEqual(
                len(self.sentences),
                len(self.tokenized_fancy_with_stopping_student),
                "Need to first pass the test: tokenization_fancy_with_stopping!",
            )

            for idx, (curr_sentence, solution_tokens, student_tokens) in tqdm(
                enumerate(
                    zip(
                        self.sentences,
                        self.tokenized_fancy_with_stopping_solution,
                        self.tokenized_fancy_with_stopping_student,
                    )
                ),
                desc="test_stemming",
            ):
                # tokenize the sentence using solution code
                golden_result = stemming(solution_tokens, self.stopwords)

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                stemming_results = self.method_wrapper(
                    stemming_method,
                    student_tokens,
                )

                self.assertion_set(
                    curr_sentence=curr_sentence,
                    golden_results=golden_result,
                    tokenized_results=stemming_results,
                )

        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    @weight(0)
    @visibility("visible")
    @number("1.7")
    def test_07_sample_tokenization_without_stopping_and_stemming(self):
        self.checker()
        try:
            # get target function reflection
            tokenizer_method = self.client.ref("tokenization")

            for idx, curr_sentence in tqdm(
                enumerate(self.sentences),
                desc="test_tokenization_no_stopping_and_stemming",
            ):
                # tokenize the sentence using solution code
                golden_results = tokenization(
                    [curr_sentence], stopwords=None, need_stemming=False
                )

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                tokenized_results = self.method_wrapper(
                    tokenizer_method, [curr_sentence], None, False
                )

                # check if sentence is tokenized into desired amount of token
                self.assertEqual(
                    len(golden_results),
                    len(tokenized_results),
                    "Output length does not match.\n"
                    + f"Expect {len(golden_results)} tokens but {len(tokenized_results)} received!\n"
                    + f"Current sentence: {curr_sentence}\n"
                    + f"Expect results: {golden_results}\n"
                    + f"but received: {tokenized_results}",
                )

                # check if sentence is tokenized into desired tokens
                self.assertEqual(
                    sorted(golden_results),
                    sorted(tokenized_results),
                    "Output does not match.\n"
                    + f"Current sentence: {curr_sentence}\n"
                    + f"Expect results: {golden_results}\n"
                    + f"but received: {tokenized_results}",
                )

        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    @weight(0)
    @visibility("visible")
    @number("1.8")
    def test_08_sample_tokenization_with_stopping_and_stemming(self):
        self.checker()
        try:
            # get target function reflection
            tokenizer_method = self.client.ref("tokenization")

            self.__class__.tokenization_solution = []
            self.__class__.tokenization_student = []

            for idx, curr_sentence in tqdm(
                enumerate(self.sentences),
                desc="test_tokenization_full",
            ):
                # tokenize the sentence using solution code
                golden_results = tokenization(
                    [curr_sentence], stopwords=self.stopwords, need_stemming=True
                )

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                tokenized_results = self.method_wrapper(
                    tokenizer_method,
                    [curr_sentence],
                    stopwords=self.stopwords,
                    need_stemming=True,
                )

                # check if sentence is tokenized into desired amount of token
                self.assertEqual(
                    len(golden_results),
                    len(tokenized_results),
                    "Output length does not match.\n"
                    + f"Expect {len(golden_results)} tokens but {len(tokenized_results)} received!\n"
                    + f"Current sentence: {curr_sentence}\n"
                    + f"Expect results: {golden_results}\n"
                    + f"but received: {tokenized_results}",
                )

                # check if sentence is tokenized into desired tokens
                self.assertEqual(
                    sorted(golden_results),
                    sorted(tokenized_results),
                    "Output does not match.\n"
                    + f"Current sentence: {curr_sentence}\n"
                    + f"Expect results: {golden_results}\n"
                    + f"but received: {tokenized_results}",
                )

                self.__class__.tokenization_solution.extend(golden_results)
                self.__class__.tokenization_student.extend(tokenized_results)
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    @weight(0)
    @visibility("visible")
    @number("1.9")
    def test_09_sample_heaps(self):
        self.checker()

        self.assertEqual(
            len(self.sentences),
            len(self.tokenized_fancy_with_stopping_student),
            "Need to first pass the test: tokenization_with_stopping_and_stemming!",
        )
        try:
            # get target function reflection
            heaps_method = self.client.ref("heaps")

            # tokenize the sentence using solution code
            golden_results = heaps(self.tokenization_solution)
            heaps_results = self.method_wrapper(
                heaps_method,
                self.tokenization_student,
            )
            # check if sentence is tokenized into desired amount of token
            self.assertEqual(
                len(golden_results),
                len(heaps_results),
                "Output length does not match.\n"
                + f"Expect {len(golden_results)} entries but {len(heaps_results)} received!\n",
            )

            # check if sentence is tokenized into desired tokens
            self.assertEqual(golden_results, heaps_results, "Output does not match.")
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    @weight(0)
    @visibility("visible")
    @number("1.10")
    def test_10_sample_statistics(self):
        self.checker()

        self.assertEqual(
            len(self.sentences),
            len(self.tokenized_fancy_with_stopping_student),
            "Need to first pass the test: tokenization_with_stopping_and_stemming!",
        )
        try:
            # get target function reflection
            statistics_method = self.client.ref("statistics")

            # tokenize the sentence using solution code
            (
                golden_token_count,
                golden_unique_token_count,
                golden_top_100_freq_tokens,
            ) = statistics(self.tokenization_solution)
            (
                student_token_count,
                student_unique_token_count,
                student_top_100_freq_tokens,
            ) = self.method_wrapper(
                statistics_method,
                self.tokenization_student,
            )

            # check if total token count matches
            self.assertEqual(
                golden_token_count,
                student_token_count,
                "Token count does not match."
                + f"Expect {golden_token_count} tokens but {student_token_count} received!\n",
            )

            # check if unique token count matches
            self.assertEqual(
                golden_unique_token_count,
                student_unique_token_count,
                "Unique token count does not match."
                + f"Expect {golden_unique_token_count} unique tokens but {student_unique_token_count} received!\n",
            )

            # check if top 100 frequent tokens match
            self.assertEqual(
                golden_top_100_freq_tokens,
                student_top_100_freq_tokens,
                "Top 100 frequent tokens do not match."
                + f"Expect: {golden_top_100_freq_tokens} ...",
            )

        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def assertion_set(
        self,
        curr_sentence: str,
        golden_results: List[Any],
        tokenized_results: List[Any],
    ) -> None:
        # check if sentence is tokenized into desired amount of token
        self.assertEqual(
            len(golden_results),
            len(tokenized_results),
            "Output length does not match.\n"
            + f"Expect {len(golden_results)} tokens but {len(tokenized_results)} received!\n"
            + f"Current sentence: {curr_sentence}\n"
            + f"Expected results: [{', '.join(golden_results)}]",
        )

        # check if sentence is tokenized into desired tokens
        self.assertEqual(
            sorted(golden_results),
            sorted(tokenized_results),
            "Output does not match.\n"
            + f"Current sentence: {curr_sentence}\n"
            + f"Expect results: [{', '.join(golden_results)}]\n"
            + f"but received: [{', '.join(tokenized_results)}]",
        )

    def tokenization_tester(
        self, function_name: str, solution_function: Callable, tqdm_desc: str = None
    ):
        if function_name == "tokenize_fancy":
            self.__class__.tokenized_fancy_solution = []
            self.__class__.tokenized_fancy_student = []

        self.checker()
        try:
            # get target function reflection
            submission_method = self.client.ref(function_name)

            for idx, curr_sentence in tqdm(enumerate(self.sentences), desc=tqdm_desc):
                # tokenize the sentence using solution code
                golden_result = solution_function(curr_sentence)

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                target_result = self.method_wrapper(submission_method, curr_sentence)

                self.assertion_set(
                    curr_sentence=curr_sentence,
                    golden_results=golden_result,
                    tokenized_results=target_result,
                )

                if function_name == "tokenize_fancy":
                    self.__class__.tokenized_fancy_solution.append(golden_result)
                    self.__class__.tokenized_fancy_student.append(target_result)

        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")
