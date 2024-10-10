from typing import Any, Callable, List

from tests import TestJupyterNotebook, heaps, load_file, statistics, tokenization, tqdm


class DebugMsgConfig(object):
    show_msg: bool = True
    test_tag: str = ""

    def __init__(self, show_msg: bool = True, test_tag: str = ""):
        self.show_msg = show_msg
        self.test_tag = test_tag


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
        overall_golden_results: List[Any] = None,
        overall_tokenized_results: List[Any] = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        expected_golden = golden_results
        if overall_golden_results is not None:
            expected_golden = overall_golden_results
        received_tokenized = tokenized_results
        if overall_tokenized_results is not None:
            received_tokenized = overall_tokenized_results
        # check if sentence is tokenized into desired amount of token
        self.assertion_wrapper(
            self.assertEqual,
            len(golden_results),
            len(tokenized_results),
            debug_msg="Output length does not match.\n"
            + f'Current sentence: "{curr_sentence}"\n'
            + f"Expect '{len(golden_results)}' tokens, "
            + f"but '{len(tokenized_results)}' received!\n"
            + f"Expect results: {expected_golden}, \n"
            + f"but received: {received_tokenized}",
            show_debug_msg=show_debug_msg,
        )

        # check if sentence is tokenized into desired tokens
        self.assertion_wrapper(
            self.assertEqual,
            sorted(golden_results),
            sorted(tokenized_results),
            debug_msg="Output does not match.\n"
            + f'Current sentence: "{curr_sentence}"\n'
            + f"Expect results: {expected_golden}, \n"
            + f"but received: {received_tokenized}",
            show_debug_msg=show_debug_msg,
        )

    def prerequisite_check(
        self,
        prerequisite: tuple[str, str] = None,
    ) -> tuple[Any, Any]:
        """
        Check whether expected class variable is stored and has the same size as self.sentences.

        If not, it means the previous test is failed which fails all downstream tasks.

        Example attr_name: "tokenize_fancy_{store_type}"
        """
        try:
            if prerequisite is None:
                return

            attr_name, func_name = prerequisite

            error_msg = f"Need to first pass the test for function(s): {func_name} !"

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
        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def save_class_attr(self, tag_name: str, results_set: tuple[Any, Any]) -> None:
        store_class_var = f"{tag_name}_{{store_type}}"
        for results, store_type in zip(results_set, ["solution", "student"]):
            setattr(
                self.__class__, store_class_var.format(store_type=store_type), results
            )

    def assertion_wrapper(
        self,
        assertion_method: Callable,
        *assertion_params,
        debug_msg: str = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        # show_debug_msg is none or show_debug_msg.show_msg is true: show the debug msg to student
        # elsewise, do not show
        if show_debug_msg is not None and not show_debug_msg.show_msg:
            try:
                assertion_method(*assertion_params, debug_msg)
            except AssertionError as e:
                prev_debug_msgs = getattr(self.__class__, "hidden_debug_msg", "")
                curr_debug_msg = "\n".join(
                    [prev_debug_msgs, "=" * 80, show_debug_msg.test_tag, "=" * 80, str(e), "=" * 80]
                )
                setattr(self.__class__, "hidden_debug_msg", curr_debug_msg)
                # print("=" * 50)
                # print(show_debug_msg.test_tag)
                # print("=" * 50)
                # print(e)
                # print("=" * 50)
                raise
            # self.assertFalse(test_failed,msg=f"Test Failed!")
        else:
            assertion_method(*assertion_params, debug_msg)

    def no_prerequisite_tester(
        self,
        function_name: str,
        solution_function: Callable,
        *function_args,
        tag_name: str = None,
        tqdm_desc: str = None,
        show_debug_msg: DebugMsgConfig = None,
        **function_kwargs,
    ) -> None:
        try:
            self.checker()
            solution_results, student_results = [], []

            # get target function reflection
            submission_method = self.client.ref(function_name)

            for idx, curr_sentence in tqdm(enumerate(self.sentences), desc=tqdm_desc):
                # tokenize the sentence using solution code
                golden_results = solution_function(
                    curr_sentence, *function_args, **function_kwargs
                )

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                target_results = self.method_wrapper(
                    submission_method, curr_sentence, *function_args, **function_kwargs
                )

                self.assertion_set(
                    curr_sentence=curr_sentence,
                    golden_results=golden_results,
                    tokenized_results=target_results,
                    show_debug_msg=show_debug_msg,
                )

                solution_results.append(golden_results)
                student_results.append(target_results)

            self.save_class_attr(
                tag_name=tag_name if tag_name is not None else function_name,
                results_set=[solution_results, student_results],
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def prerequisite_tester(
        self,
        function_name: str,
        solution_function: Callable,
        *function_args,
        tag_name: str = None,
        tqdm_desc: str = None,
        prerequisite: tuple[str, str] = None,
        input_is_list: bool = False,
        show_debug_msg: DebugMsgConfig = None,
        **function_kwargs,
    ) -> None:
        try:
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
                if not input_is_list:
                    # tokenize the sentence using solution code
                    curr_solution_results = solution_function(
                        solution_tokens, *function_args, **function_kwargs
                    )
                    # cleanup the notebook output as ipython core will give out warning
                    # when output has 200? more line and corrupts reflection of the function output
                    self.clear_notebook_output(idx)

                    # check the output of notebook
                    curr_students_results = self.method_wrapper(
                        student_method,
                        student_tokens,
                        *function_args,
                        **function_kwargs,
                    )

                    self.assertion_set(
                        curr_sentence=curr_sentence,
                        golden_results=curr_solution_results,
                        tokenized_results=curr_students_results,
                        show_debug_msg=show_debug_msg,
                    )

                else:
                    curr_solution_results = []
                    curr_students_results = []
                    for solution_token, student_token in zip(
                        solution_tokens, student_tokens
                    ):
                        # tokenize the sentence using solution code
                        curr_solution_result = solution_function(
                            solution_token, *function_args, **function_kwargs
                        )

                        # cleanup the notebook output as ipython core will give out warning
                        # when output has 200? more line and corrupts reflection of the function output
                        self.clear_notebook_output(idx)

                        # check the output of notebook
                        curr_students_result = self.method_wrapper(
                            student_method,
                            student_token,
                            *function_args,
                            **function_kwargs,
                        )

                        self.assertion_set(
                            curr_sentence=curr_sentence,
                            golden_results=curr_solution_result,
                            tokenized_results=curr_students_result,
                            show_debug_msg=show_debug_msg,
                        )
                        curr_solution_results.append(curr_solution_result)
                        curr_students_results.append(curr_students_result)

                solution_results.append(curr_solution_results)
                student_results.append(curr_students_results)

            self.save_class_attr(
                tag_name=tag_name if tag_name is not None else function_name,
                results_set=[solution_results, student_results],
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def tokenization_tester(
        self,
        tag_name: str,
        stopwords: list[str],
        tokenizer_type: str,
        stemming_type: str,
        tqdm_desc: str = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        try:
            self.checker()
            # get target function reflection
            tokenization_method = self.client.ref("tokenization")

            solution_results, student_results = [], []

            for idx, curr_sentence in tqdm(
                enumerate(self.sentences),
                desc=tqdm_desc,
            ):
                # tokenize the sentence using solution code
                golden_results = tokenization(
                    [curr_sentence],
                    stopwords=stopwords,
                    tokenizer_type=tokenizer_type,
                    stemming_type=stemming_type,
                )

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                tokenized_results = self.method_wrapper(
                    tokenization_method,
                    [curr_sentence],
                    stopwords=stopwords,
                    tokenizer_type=tokenizer_type,
                    stemming_type=stemming_type,
                )

                for (curr_golden_token, curr_golden_subtokens), (
                    curr_student_token,
                    curr_student_subtokens,
                ) in zip(golden_results, tokenized_results):
                    # Check each subtoken list, sort the list by alphabetic order

                    # check if sentence is tokenized into desired amount of token
                    self.assertion_wrapper(
                        self.assertEqual,
                        curr_golden_token,
                        curr_student_token,
                        debug_msg="(main) Token mismatch.\n"
                        + f"Expect token '{curr_golden_token}', but '{curr_student_token}' received!",
                        show_debug_msg=show_debug_msg,
                    )

                    # check if sentence is tokenized into desired tokens
                    self.assertion_wrapper(
                        self.assertEqual,
                        sorted(curr_golden_subtokens),
                        sorted(curr_student_subtokens),
                        debug_msg="Output does not match.\n"
                        + f'Current sentence: "{curr_sentence}"\n'
                        + f"Expect results: {curr_golden_subtokens}\n"
                        + f"but received: {curr_student_subtokens}",
                        show_debug_msg=show_debug_msg,
                    )

                solution_results.extend(golden_results)
                student_results.extend(tokenized_results)

            self.save_class_attr(
                tag_name=tag_name, results_set=[solution_results, student_results]
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def heaps_tester(
        self,
        prerequisite: tuple[str, str],
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        try:
            self.checker()

            prev_solution_results, prev_students_results = self.prerequisite_check(
                prerequisite=prerequisite,
            )
            # get target function reflection
            heaps_method = self.client.ref("heaps")

            # tokenize the sentence using solution code
            golden_results = heaps(prev_solution_results)
            heaps_results = self.method_wrapper(
                heaps_method,
                prev_students_results,
            )
            # check if sentence is tokenized into desired amount of token
            self.assertion_wrapper(
                self.assertEqual,
                len(golden_results),
                len(heaps_results),
                debug_msg="Output length does not match.\n"
                + f"Expect '{len(golden_results)}' entries but '{len(heaps_results)}' received!\n"
                + f"Expect results: {golden_results}\n"
                + f"but received: {heaps_results}",
                show_debug_msg=show_debug_msg,
            )

            if golden_results[-1] != heaps_results[-1]:
                golden_token_count, golden_unique_token_count = golden_results[-1]
                student_token_count, student_unique_token_count = heaps_results[-1]
                # check if total token count matches
                self.assertion_wrapper(
                    self.assertEqual,
                    golden_token_count,
                    student_token_count,
                    debug_msg="Token count does not match.\n"
                    + f"Expect '{golden_token_count}' tokens but '{student_token_count}' received!\n",
                    show_debug_msg=show_debug_msg,
                )

                # check if unique token count matches
                self.assertion_wrapper(
                    self.assertEqual,
                    golden_unique_token_count,
                    student_unique_token_count,
                    debug_msg="Unique token count does not match.\n"
                    + f"Expect '{golden_unique_token_count}' unique tokens but '{student_unique_token_count}' received!\n",
                    show_debug_msg=show_debug_msg,
                )

        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def zipf_tester(
        self,
        prerequisite: tuple[str, str],
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        try:
            self.checker()
            prev_solution_results, prev_students_results = self.prerequisite_check(
                prerequisite=prerequisite,
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
            self.assertion_wrapper(
                self.assertEqual,
                golden_token_count,
                student_token_count,
                debug_msg="Token count does not match.\n"
                + f"Expect '{golden_token_count}' tokens but '{student_token_count}' received!\n",
                show_debug_msg=show_debug_msg,
            )

            # check if unique token count matches
            self.assertion_wrapper(
                self.assertEqual,
                golden_unique_token_count,
                student_unique_token_count,
                debug_msg="Unique token count does not match.\n"
                + f"Expect '{golden_unique_token_count}' unique tokens but '{student_unique_token_count}' received!\n",
                show_debug_msg=show_debug_msg,
            )

            golden_top_100_freq_tokens.sort(key=lambda x: (x[1], x[0]))
            student_top_100_freq_tokens.sort(key=lambda x: (x[1], x[0]))
            # check if top 100 frequent tokens match
            for curr_golden, curr_student in zip(
                golden_top_100_freq_tokens,
                student_top_100_freq_tokens,
            ):
                # check if heaps number matches
                self.assertion_wrapper(
                    self.assertEqual,
                    list(curr_golden),
                    list(curr_student),
                    debug_msg="Top 100 frequent tokens do not match.\n"
                    + f"Expect {list(curr_golden)} but {list(curr_student)} received!\n"
                    + f"Expect results: {golden_top_100_freq_tokens}\n"
                    + f"but received: {student_top_100_freq_tokens}",
                    show_debug_msg=show_debug_msg,
                )
        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")
