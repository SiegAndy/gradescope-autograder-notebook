from typing import Any, Callable, List, Tuple

from gradescope_utils.autograder_utils.decorators import number, visibility, weight
from tqdm import tqdm

from tests import (
    TestJupyterNotebook,
    heaps,
    statistics,
    stemming_porter,
    stemming_s,
    stopping,
    tokenization,
    tokenize_4grams,
    tokenize_fancy,
    tokenize_space,
)


class TestNotebookCompilable(TestJupyterNotebook):

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            jupyter_notebook_file_path="test3.ipynb",
            # data_file_path="test2",
            data_file_path="P1-train.gz",
            allowed_imports=["re"],
            # allowed_imports=["gzip", "pathlib"],
        )
        return

    @weight(0)
    @visibility("visible")
    @number("0.1.1")
    def test_11_ipynb_compilable_and_packages(self):
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("0.2.1")
    def test_21_sample_tokenize_space(self):
        self.no_prerequisite_tester(
            function_name="tokenize_space",
            solution_function=tokenize_space,
            tag_name="tokenized_space",
            tqdm_desc="test_tokenize_space",
        )

    @weight(0)
    @visibility("visible")
    @number("0.2.2")
    def test_22_sample_tokenize_4grams(self):
        self.no_prerequisite_tester(
            function_name="tokenize_4grams",
            solution_function=tokenize_4grams,
            tag_name="tokenized_4grams",
            tqdm_desc="test_tokenize_4grams",
        )

    @weight(0)
    @visibility("visible")
    @number("0.2.3")
    def test_23_sample_tokenize_fancy(self):
        self.no_prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_fancy",
            tqdm_desc="test_tokenize_fancy",
        )

    @weight(0)
    @visibility("visible")
    @number("0.3.1")
    def test_31_sample_tokenize_fancy_yesStopping(self):
        self.prerequisite_tester(
            function_name="stopping",
            solution_function=stopping,
            tag_name="tokenized_fancy_yesStopping",
            tqdm_desc="test_stopping",
            prerequisite=("tokenized_fancy_{store_type}", "tokenize_fancy"),
            stopwords = self.stopwords,
        )

    @weight(0)
    @visibility("visible")
    @number("0.4.1")
    def test_41_sample_tokenize_space_noStopping_and_stemming_s(self):
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenized_space_noStopping_and_stemming_s",
            tqdm_desc="test_tokenize_space_noStopping_and_stemming_s",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(0)
    @visibility("visible")
    @number("0.4.2")
    def test_42_sample_tokenize_space_noStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenized_space_noStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_space_noStopping_and_stemming_porter",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )


    @weight(0)
    @visibility("visible")
    @number("0.4.3")
    def test_43_sample_tokenize_fancy_yesStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_fancy_yesStopping_and_stemming_porter",
            prerequisite=("tokenized_fancy_yesStopping_{store_type}", "tokenize_fancy, stopping"),
        )

    @weight(0)
    @visibility("visible")
    @number("0.5.1")
    def test_51_sample_tokenization_fancy_yesStopping_and_stemming_porter(self):
        self.checker()
        # get target function reflection
        tokenization_method = self.client.ref("tokenization")

        solution_results, student_results = [], []

        for idx, curr_sentence in tqdm(
            enumerate(self.sentences),
            desc="test_tokenization_full",
        ):
            # tokenize the sentence using solution code
            golden_results = tokenization(
                [curr_sentence],
                stopwords=self.stopwords,
                tokenizer_type="fancy",
                stemming_type="porter",
            )

            # cleanup the notebook output as ipython core will give out warning
            # when output has 200? more line and corrupts reflection of the function output
            self.clear_notebook_output(idx)

            # check the output of notebook
            tokenized_results = self.method_wrapper(
                tokenization_method,
                [curr_sentence],
                stopwords=self.stopwords,
                tokenizer_type="fancy",
                stemming_type="porter",
            )

            for (curr_golden_token, curr_golden_subtokens), (
                curr_student_token,
                curr_student_subtokens,
            ) in zip(golden_results, tokenized_results):
                # Check each subtoken list, sort the list by alphabetic order

                # check if sentence is tokenized into desired amount of token
                self.assertEqual(
                    curr_golden_token,
                    curr_student_token,
                    "(main) Token mismatch.\n"
                    + f"Expect token '{curr_golden_token}, "
                    + f"but '{curr_student_token}' received!\n",
                )

                # check if sentence is tokenized into desired tokens
                self.assertEqual(
                    sorted(curr_golden_subtokens),
                    sorted(curr_student_subtokens),
                    "Output does not match.\n"
                    + f'Current sentence: "{curr_sentence}"\n'
                    + f"Expect results: {curr_golden_subtokens}\n"
                    + f"but received: {curr_student_subtokens}",
                )

            solution_results.extend(golden_results)
            student_results.extend(tokenized_results)
        
        self.save_class_attr(
            tag_name="tokenization_fancy_yesStopping_and_stemming_porter",
            results_set=[solution_results, student_results]
        )
        
        
    @weight(0)
    @visibility("visible")
    @number("0.6.1")
    def test_61_sample_heaps(self):
        self.checker()

        prev_solution_results, prev_students_results = self.prerequisite_check(
            prerequisite=("tokenization_fancy_yesStopping_and_stemming_porter_{store_type}", "tokenization(tokenize_type=\"fancy\", stopwords=stopwords, stemming_type=\"porter\")"),
        )
        # self.assertEqual(
        #     len(self.sentences),
        #     len(self.tokenized_fancy_with_stopping_student),
        #     "Need to first pass the test: tokenization_yesStopping_and_stemming_porter!",
        # )
        # get target function reflection
        heaps_method = self.client.ref("heaps")

        # tokenize the sentence using solution code
        golden_results = heaps(prev_solution_results)
        heaps_results = self.method_wrapper(
            heaps_method,
            prev_students_results,
        )
        # check if sentence is tokenized into desired amount of token
        self.assertEqual(
            len(golden_results),
            len(heaps_results),
            "Output length does not match.\n"
            + f"Expect {len(golden_results)} entries but {len(heaps_results)} received!\n",
        )

        for curr_golden, curr_student in zip(golden_results, heaps_results):
            # check if heaps number matches
            self.assertEqual(
                list(curr_golden),
                list(curr_student),
                "Output does not match.\n"
                + f"Expect results: {list(curr_golden)}\n"
                + f"but received: {list(curr_student)}",
            )

    @weight(0)
    @visibility("visible")
    @number("0.7.1")
    def test_71_sample_statistics(self):
        self.checker()
        prev_solution_results, prev_students_results = self.prerequisite_check(
            prerequisite=("tokenization_fancy_yesStopping_and_stemming_porter_{store_type}", "tokenization(tokenize_type=\"fancy\", stopwords=stopwords, stemming_type=\"porter\")"),
        )
        # get target function reflection
        statistics_method = self.client.ref("statistics")

        # tokenize the sentence using solution code
        (
            golden_token_count,
            golden_unique_token_count,
            golden_top_100_freq_tokens,
        ) = statistics(prev_solution_results)
        (
            student_token_count,
            student_unique_token_count,
            student_top_100_freq_tokens,
        ) = self.method_wrapper(
            statistics_method,
            prev_students_results,
        )

        # check if total token count matches
        self.assertEqual(
            golden_token_count,
            student_token_count,
            "Token count does not match.\n"
            + f"Expect {golden_token_count} tokens but {student_token_count} received!\n",
        )

        # check if unique token count matches
        self.assertEqual(
            golden_unique_token_count,
            student_unique_token_count,
            "Unique token count does not match.\n"
            + f"Expect {golden_unique_token_count} unique tokens but {student_unique_token_count} received!\n",
        )

        # check if top 100 frequent tokens match
        for curr_golden, curr_student in zip(
            sorted(golden_top_100_freq_tokens, key=lambda x: (x[1], x[0])),
            sorted(student_top_100_freq_tokens, key=lambda x: (x[1], x[0])),
        ):
            # check if heaps number matches
            self.assertEqual(
                list(curr_golden),
                list(curr_student),
                "Top 100 frequent tokens do not match.\n"
                + f"Expect result: {list(curr_golden)}\n"
                + f"but received: {list(curr_student)}",
            )

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
            + f'Current sentence: "{curr_sentence}"\n'
            + f"Expect '{len(golden_results)}' tokens, "
            + f"but '{len(tokenized_results)}' received!\n"
        )

        # check if sentence is tokenized into desired tokens
        self.assertEqual(
            sorted(golden_results),
            sorted(tokenized_results),
            "Output does not match.\n"
            + f'Current sentence: "{curr_sentence}"\n'
            + f"Expect results: {golden_results}, \n"
            + f"but received: {tokenized_results}",
        )

    def prerequisite_check(
        self, prerequisite: tuple[str, str] = None
    ) -> tuple[Any, Any]:
        """
        Check whether expected class variable is stored and has the same size as self.sentences.

        If not, it means the previous test is failed which fails all downstream tasks.

        Example attr_name: "tokenize_fancy_{store_type}"
        """
        if prerequisite is None:
            return

        attr_name, func_name = prerequisite

        error_msg = f"Need to first pass the test for function(s): {func_name}!"

        result_attrs = []
        for store_type in ["solution", "student"]:
            curr_attr_name = attr_name.format(store_type=store_type)
            # check if class has the attribute
            self.assertTrue(
                hasattr(self.__class__, curr_attr_name),
                error_msg,
            )

            curr_attr = getattr(self.__class__, curr_attr_name)

            # check if the attribute is None
            self.assertIsNotNone(
                curr_attr,
                error_msg,
            )
            if "tokenization" not in curr_attr_name:
                # only check whether the attribute has the same size as self.sentences
                # when the checker is called for tokenizer, stemmer or stopper function.
                self.assertEqual(
                    len(self.sentences),
                    len(curr_attr),
                    error_msg + f"\t{curr_attr_name}",
                )

            result_attrs.append(curr_attr)
        return result_attrs

    def save_class_attr(self, tag_name: str, results_set: tuple[Any, Any]):
        store_class_var = f"{tag_name}_{{store_type}}"
        for results, store_type in zip(
            results_set, ["solution", "student"]
        ):
            setattr(
                self.__class__, store_class_var.format(store_type=store_type), results
            )
    
    def no_prerequisite_tester(
        self,
        function_name: str,
        solution_function: Callable,
        *function_args,
        tag_name: str = None,
        tqdm_desc: str = None,
        **function_kwargs,
    ):
        self.checker()
        solution_results, student_results = [], []

        # get target function reflection
        submission_method = self.client.ref(function_name)

        for idx, curr_sentence in tqdm(enumerate(self.sentences), desc=tqdm_desc):
            # tokenize the sentence using solution code
            golden_results = solution_function(curr_sentence, *function_args, **function_kwargs)

            # cleanup the notebook output as ipython core will give out warning
            # when output has 200? more line and corrupts reflection of the function output
            self.clear_notebook_output(idx)

            # check the output of notebook
            target_results = self.method_wrapper(submission_method, curr_sentence, *function_args, **function_kwargs)

            self.assertion_set(
                curr_sentence=curr_sentence,
                golden_results=golden_results,
                tokenized_results=target_results,
            )

            solution_results.append(golden_results)
            student_results.append(target_results)

        self.save_class_attr(
            tag_name=tag_name if tag_name is not None else function_name,
            results_set=[solution_results, student_results]
        )
        
    def prerequisite_tester(
        self,
        function_name: str,
        solution_function: Callable,
        *function_args,
        tag_name: str = None,
        tqdm_desc: str = None,
        prerequisite: tuple[str, str] = None,
        **function_kwargs,
    ):
        self.checker()
        prev_solution_results, prev_students_results = self.prerequisite_check(
            prerequisite=prerequisite
        )
        # get target function reflection
        student_method = self.client.ref(function_name)
        solution_results, student_results = [], []

        for idx, (curr_sentence, solution_tokens, student_tokens) in tqdm(
            enumerate(
                zip(
                    self.sentences,
                    prev_solution_results,
                    prev_students_results,
                )
            ),
            desc=tqdm_desc,
        ):
            # tokenize the sentence using solution code
            curr_solution_results = solution_function(
                solution_tokens, *function_args, **function_kwargs
            )

            # cleanup the notebook output as ipython core will give out warning
            # when output has 200? more line and corrupts reflection of the function output
            self.clear_notebook_output(idx)

            # check the output of notebook
            curr_students_results = self.method_wrapper(
                student_method, student_tokens, *function_args, **function_kwargs
            )

            self.assertion_set(
                curr_sentence=curr_sentence,
                golden_results=curr_solution_results,
                tokenized_results=curr_students_results,
            )

            solution_results.append(curr_solution_results)
            student_results.append(curr_students_results)

        self.save_class_attr(
            tag_name=tag_name if tag_name is not None else function_name,
            results_set=[solution_results, student_results]
        )
        
