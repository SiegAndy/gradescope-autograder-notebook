from typing import Any, Callable, List

from tqdm import tqdm

from tests import TestJupyterNotebook, load_file


class TestPA1(TestJupyterNotebook):
    sentences: List[str]  # loaded sentence from dataset
    stopwords: List[str]  # loaded stopwords from dataset
    allowed_imports: List[str]

    @classmethod
    def setUpClass(
        cls,
        data_file_path: str,
        allowed_imports: List[str] = None,
    ):
        cls.allowed_imports = allowed_imports
        cls.sentences = load_file(
            data_file_path,
            gz_zip=True if data_file_path.endswith(".gz") else False,
        )
        cls.stopwords = load_file(
            "stopwords.txt",
            gz_zip=False,
        )
        super().setUpClass()    

    
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
        
