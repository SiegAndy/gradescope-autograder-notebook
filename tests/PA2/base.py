from collections import defaultdict
from typing import Any, Callable, List

from tests.base import (
    DebugMsgConfig,
    TestJupyterNotebook,
    tqdm,
)

from tests.PA2.solution import (
    autograder_version,
    load_file,
    freq_stats,
    preprocessing,
)


class TestPA2(TestJupyterNotebook):
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
        super().setUpClass(autograder_version=autograder_version)

    def assertion_set(
        self,
        curr_sentence: str,
        golden_results: List[Any],
        processed_results: List[Any],
        overall_golden_results: List[Any] = None,
        overall_tokenized_results: List[Any] = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:

        expected_golden = golden_results
        if overall_golden_results is not None:
            expected_golden = overall_golden_results

        received_tokenized = processed_results
        if overall_tokenized_results is not None:
            received_tokenized = overall_tokenized_results

        # check if sentence is tokenized into desired amount of token
        self.assertion_wrapper(
            self.assertEqual,
            len(golden_results),
            len(processed_results),
            debug_msg="Output length does not match.\n"
            + f'Current Sentence: "{curr_sentence}"\n'
            + f"Expect '{len(golden_results)}' tokens, "
            + f"but '{len(processed_results)}' received!\n"
            + f"Expected Outputs: '{expected_golden}'\n"
            + f"Received Outputs: '{received_tokenized}'",
            show_debug_msg=show_debug_msg,
        )

        # check if sentence is tokenized into desired tokens
        self.assertion_wrapper(
            self.assertEqual,
            golden_results,
            processed_results,
            debug_msg="Output does not match.\n"
            + f'Current Sentence: "{curr_sentence}"\n'
            + f"Expected Outputs: '{expected_golden}'\n"
            + f"Received Outputs: '{received_tokenized}'",
            show_debug_msg=show_debug_msg,
        )

    def prerequisite_check(
        self,
        prerequisite_fn_tags: list[str] = None,
        prerequisite_fn_names: list[str] = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> tuple[Any, Any]:
        """
        Check whether expected class variable is stored and has the same size as self.sentences.

        If not, it means the previous test is failed which fails all downstream tasks.

        prerequisite_fn_tags are the test tags that we used to check whether previous tests are pass or fail.

        prerequisite_fn_names are the function names that are testing on previous test.

        Example prerequisite_fn_tags: ["sample_tokenize_fancy"]

        Example prerequisite_fn_names: ["tokenize_fancy"]
        """
        try:
            if len(prerequisite_fn_tags) == 0 or len(prerequisite_fn_names) == 0:
                return

            error_msg = f"Need to first pass the test(s) for function(s): {', '.join([fn_name +'()' if not fn_name.endswith(')') else fn_name for fn_name in prerequisite_fn_names] )} !"

            result_attrs = defaultdict(dict)
            for test_tag in prerequisite_fn_tags:
                for store_type in ["solution", "student"]:
                    curr_attr_name = f"{test_tag}_{store_type}"
                    # check if class has the attribute
                    self.assertion_wrapper(
                        self.assertTrue,
                        hasattr(self.__class__, curr_attr_name),
                        debug_msg=error_msg,
                        show_debug_msg=show_debug_msg,
                    )

                    curr_attr = getattr(self.__class__, curr_attr_name)

                    # check if the attribute is None
                    self.assertion_wrapper(
                        self.assertIsNotNone,
                        curr_attr,
                        debug_msg=error_msg,
                        show_debug_msg=show_debug_msg,
                    )

                    result_attrs[test_tag][store_type] = curr_attr
            return result_attrs
        except AssertionError:
            raise
        except Exception as e:
            self.assertion_wrapper(
                self.assertTrue,
                False,
                debug_msg=f"Code does not compile:\n{e}",
                show_debug_msg=show_debug_msg,
            )

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
        show_debug_msg = None  # suppress error managements
        if show_debug_msg is not None and not show_debug_msg.show_msg_in_orig_test:
            try:
                assertion_method(*assertion_params, debug_msg)
            except AssertionError as e:
                prev_debug_msgs = getattr(self.__class__, "hidden_debug_msg", "")
                curr_debug_msg = "\n".join(
                    [
                        prev_debug_msgs,
                        "=" * 80,
                        show_debug_msg.test_tag,
                        "=" * 80,
                        str(e),
                        "=" * 80,
                    ]
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
        solution_function: Callable,
        *function_args,
        tag_name: str = None,
        tqdm_desc: str = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
        **function_kwargs,
    ) -> None:
        try:
            self.checker()
            solution_results, student_results = [], []
            function_name = solution_function.__name__

            # get target function reflection
            submission_method = self.client.ref(function_name)

            for idx, curr_sentence in tqdm(
                enumerate(self.sentences), total=len(self.sentences), desc=tqdm_desc
            ):
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
                    processed_results=target_results,
                    show_debug_msg=show_debug_msg,
                )

                set_score(round((idx + 1) * max_score / len(self.sentences), 2))

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
        solution_function: Callable,
        *function_args,
        tag_name: str = None,
        tqdm_desc: str = None,
        prerequisite_fn_tags: list[str] = None,
        prerequisite_fn_names: list[str] = None,
        use_prev_results: bool = False,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
        **function_kwargs,
    ) -> None:
        try:
            self.checker()
            prev_results = self.prerequisite_check(
                prerequisite_fn_tags=prerequisite_fn_tags,
                prerequisite_fn_names=prerequisite_fn_names,
                show_debug_msg=show_debug_msg,
            )

            function_name = solution_function.__name__
            # get target function reflection
            student_method = self.client.ref(function_name)
            solution_results, student_results = [], []

            if not use_prev_results and len(prev_results) != 0:
                test_contents = self.sentences
            elif use_prev_results and len(prev_results) == 1:
                test_contents = zip(
                    self.sentences,
                    prev_results[prerequisite_fn_tags[0]]["solution"],
                    prev_results[prerequisite_fn_tags[0]]["student"],
                )
            else:
                if use_prev_results:
                    error_msg_header = "Unexpected params for prerequisite_tester(), should only specified one prerequisite:"
                else:
                    error_msg_header = "Unexpected params for prerequisite_tester(), should use no_prerequisite_tester() as no prerequisite_tester is specified:"
                self.assertion_wrapper(
                    self.assertTrue,
                    False,
                    debug_msg=(
                        error_msg_header
                        + f"\tprerequisite_fn_tags: {prerequisite_fn_tags}"
                        + f"\tprerequisite_fn_names: {prerequisite_fn_names}"
                    ),
                )

            for idx, curr_test_content in tqdm(
                enumerate(test_contents),
                total=len(self.sentences),
                desc=tqdm_desc,
            ):
                # tokenizer always takes in one sentence at a time and return a list of tokens
                # stopping and stemming takes in a list of tokens and output a list of tokens
                # preprocessing takes in a list of sentence and output a list of list of tokens

                if not use_prev_results and len(prev_results) != 0:
                    # not using stored result, using sentence
                    curr_sentence, prev_solution_result, prev_students_result = (
                        curr_test_content,
                        curr_test_content,
                        curr_test_content,
                    )
                elif use_prev_results and len(prev_results) == 1:
                    # using stored result
                    curr_sentence, prev_solution_result, prev_students_result = (
                        curr_test_content
                    )
                # tokenize the sentence using solution code
                curr_solution_results = solution_function(
                    prev_solution_result, *function_args, **function_kwargs
                )
                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                curr_students_results = self.method_wrapper(
                    student_method,
                    prev_students_result,
                    *function_args,
                    **function_kwargs,
                )

                self.assertion_set(
                    curr_sentence=curr_sentence,
                    golden_results=curr_solution_results,
                    processed_results=curr_students_results,
                    show_debug_msg=show_debug_msg,
                )

                set_score(round((idx + 1) * max_score / len(self.sentences), 2))

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

    def preprocessing_tester(
        self,
        stopwords: list[str],
        tokenizer_type: str,
        stemming_type: str,
        tag_name: str,
        num_of_sample_per_batch: int = 2,
        tqdm_desc: str = None,
        prerequisite_fn_tags: list[str] = None,
        prerequisite_fn_names: list[str] = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        try:
            self.checker()

            self.prerequisite_check(
                prerequisite_fn_tags=prerequisite_fn_tags,
                prerequisite_fn_names=prerequisite_fn_names,
                show_debug_msg=show_debug_msg,
            )

            # get target function reflection
            preprocessing_method = self.client.ref("preprocessing")

            solution_results, student_results = [], []

            batch_size = num_of_sample_per_batch
            pbar = tqdm(total=len(self.sentences), desc=tqdm_desc)
            for idx in range(0, len(self.sentences), batch_size):
                curr_test_sentences = self.sentences[idx : idx + batch_size]

                # tokenize the sentence using solution code
                golden_results = preprocessing(
                    curr_test_sentences,
                    tokenizer_type=tokenizer_type,
                    stopwords=stopwords,
                    stemming_type=stemming_type,
                )

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(idx)

                # check the output of notebook
                tokenized_results = self.method_wrapper(
                    preprocessing_method,
                    curr_test_sentences,
                    tokenizer_type=tokenizer_type,
                    stopwords=stopwords,
                    stemming_type=stemming_type,
                )

                self.assertion_wrapper(
                    self.assertEqual,
                    len(golden_results),
                    len(tokenized_results),
                    debug_msg="Output length does not match on the subset:\n\t"
                    + "\n\t".join([f'"{sentence}"' for sentence in curr_test_sentences])
                    + "\n"
                    + f"Expect '{len(golden_results)}' lists of tokens, "
                    + f"but only '{len(tokenized_results)}' received!\n",
                    show_debug_msg=show_debug_msg,
                )

                for curr_test_sentence, curr_golden_tokens, curr_student_tokens in zip(
                    curr_test_sentences, golden_results, tokenized_results
                ):
                    # check if sentence is tokenized into desired tokens
                    self.assertion_wrapper(
                        self.assertEqual,
                        curr_golden_tokens,
                        curr_student_tokens,
                        debug_msg="Token mismatch.\n"
                        + f"Current Sentence: {curr_test_sentence}"
                        + f"Expected Outputs: '{curr_golden_tokens}'\n"
                        + f"Received Outputs: '{curr_student_tokens}'",
                        show_debug_msg=show_debug_msg,
                    )
                set_score(
                    round(
                        min(idx + batch_size, len(self.sentences))
                        * max_score
                        / len(self.sentences),
                        2,
                    )
                )

                if hasattr(pbar, "update"):
                    pbar.update(len(curr_test_sentences))
                solution_results.extend(golden_results)
                student_results.extend(tokenized_results)

            self.save_class_attr(
                tag_name=tag_name, results_set=[solution_results, student_results]
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")

    def zipf_tester(
        self,
        tag_name: str,
        prerequisite_fn_tags: list[str] = None,
        prerequisite_fn_names: list[str] = None,
        set_score: Callable = None,
        max_score: int = None,
        tqdm_desc: str = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        try:
            self.checker()
            prev_results = self.prerequisite_check(
                prerequisite_fn_tags=prerequisite_fn_tags,
                prerequisite_fn_names=prerequisite_fn_names,
                show_debug_msg=show_debug_msg,
            )
            # get target function reflection
            statistics_method = self.client.ref("freq_stats")
            prev_solution_results = prev_results[prerequisite_fn_tags[0]]["solution"]
            prev_student_results = prev_results[prerequisite_fn_tags[0]]["student"]

            # tokenize the sentence using solution code
            golden_results = freq_stats(prev_solution_results)
            student_results = self.method_wrapper(
                statistics_method,
                prev_student_results,
            )

            # check if total token count matches
            self.assertion_wrapper(
                self.assertEqual,
                len(golden_results),
                len(student_results),
                debug_msg="The number of Unique Token does not match.\n"
                + f"Expect '{len(golden_results)}' unique tokens but '{len(student_results)}' received!\n",
                show_debug_msg=show_debug_msg,
            )

            sorted_golden_results = sorted(golden_results, key=lambda x: (-x[1], x[0]))
            sorted_student_results = sorted(
                student_results, key=lambda x: (-x[1], x[0])
            )

            for (
                idx,
                ((golden_token, golden_token_cnt), (student_token, student_token_cnt)),
            ) in tqdm(
                enumerate(zip(sorted_golden_results, sorted_student_results)),
                desc=tqdm_desc,
            ):
                # check if unique token count matches
                if (
                    golden_token != student_token
                    or golden_token_cnt != student_token_cnt
                ):
                    self.assertion_wrapper(
                        self.assertEqual,
                        (golden_token, golden_token_cnt),
                        (student_token, student_token_cnt),
                        debug_msg="Token does not match.\n"
                        + f"Expect '{(golden_token, golden_token_cnt)}' but '{(student_token, student_token_cnt)}' received!\n"
                        + f"Expected Output: '{sorted_golden_results}'\n"
                        + f"Received Output: '{sorted_student_results}'",
                        show_debug_msg=show_debug_msg,
                    )
                set_score(round((idx + 1) * max_score / len(self.sentences), 2))

        except AssertionError:
            raise
        except Exception as e:
            self.assertTrue(False, f"Code does not compile:\n{e}")
