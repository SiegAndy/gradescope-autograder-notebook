from collections import defaultdict
from typing import Any, Callable, List

from tests.base import (
    DebugMsgConfig,
    TestJupyterNotebook,
    tqdm,
)

from tests.PA3.solution import (
    autograder_version,
    InvertedIndex,
    download_file,
    build_inverted_index,
)


class TestPA3(TestJupyterNotebook):
    solution_inverted_index: InvertedIndex
    queries: list[str]
    documents: list[dict[str, str]]
    allowed_imports: List[str]
    save_solution_flag: bool

    @classmethod
    def setUpClass(
        cls,
        test_type: str,
        level_1_limit: int = 100,
        level_2_limit: int = 10,
        allowed_imports: List[str] = None,
    ):
        cls.save_solution_flag = False
        cls.allowed_imports = allowed_imports
        public_tests_docs, public_tests_queries = download_file(
            "P3-data.zip",
        )

        # TODO: Have a real dataset with valid queries and protected docs
        protected_tests_docs, protected_tests_queries = (
            public_tests_docs,
            public_tests_queries,
        )
        # cls.queries = (
        #     public_tests_queries if "public" in test_type else protected_tests_queries
        # )

        cls.queries = {0: "of aa", 1: "yes and no"}
        cls.documents = (
            public_tests_docs if "public" in test_type else protected_tests_docs
        )

        cls.level_1_limit = level_1_limit
        cls.level_2_limit = level_2_limit

        cls.solution_inverted_index = build_inverted_index(cls.documents)
        super().setUpClass(autograder_version=autograder_version)

    def save_class_attr(
        self,
        tag_name: str,
        results_set: tuple[Any, Any],
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        store_class_var = f"{tag_name}_{{store_type}}"
        for results, store_type in zip(results_set, ["solution", "student"]):
            setattr(
                self.__class__, store_class_var.format(store_type=store_type), results
            )

            # not saving the solution
            if not self.save_solution_flag:
                continue

            in_gradescope = True if os.environ.get("in_gradescope", False) else False
            if not in_gradescope and store_type != "solution" or show_debug_msg is None:
                continue
            record_tag = show_debug_msg.test_tag
            record_folder = "./data/2025-Spring-P2/data/records/"
            import os, json

            os.makedirs(record_folder, exist_ok=True)
            curr_record_file = os.path.join(record_folder, tag_name + ".json")
            with open(curr_record_file, "w", encoding="utf-8") as f:
                json.dump({record_tag: results}, f, indent=4)

    def assertion_wrapper(
        self,
        assertion_method: Callable,
        *assertion_params,
        debug_msg: str = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        # show_debug_msg is none or show_debug_msg.show_msg is true: show the debug msg to student
        # elsewise, do not show
        # show_debug_msg = None  # suppress error managements
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
                raise
        else:
            assertion_method(*assertion_params, debug_msg)

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

    def inverted_index_parsing_tester(
        self,
        show_debug_msg: DebugMsgConfig = None,
    ):

        try:
            tag_name = "parse_inverted_index"
            self.checker(show_debug_msg=show_debug_msg)

            self.client.inject(
                f"""
                    public_tests_docs, public_tests_queries = download_file(
                        "P3-data.zip",
                    )
                    student_inverted_index = build_inverted_index(public_tests_docs)
                """,
                pop=True,
            )

            student_inverted_index: InvertedIndex = self.client.ref(
                "student_inverted_index"
            )

            self.assertion_wrapper(
                self.assertIsNotNone,
                student_inverted_index,
                debug_msg="The Parsed Inverted Index is None.\n",
                show_debug_msg=show_debug_msg,
            )

            self.save_class_attr(
                tag_name=tag_name,
                results_set=[self.solution_inverted_index, student_inverted_index],
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertion_wrapper(
                self.assertTrue,
                False,
                debug_msg=f"Code does not compile:\n{e}",
                show_debug_msg=show_debug_msg,
            )

    def tf_tester(
        self,
        solution_inverted_index: InvertedIndex,
        student_inverted_index: InvertedIndex,
        tqdm_desc: str = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        student_method = self.client.ref("tf")
        total_cnt = 0
        mismatches = []

        top_terms = sorted(
            solution_inverted_index.postings_lists.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )[: self.level_1_limit]

        for curr_term, postinglist in tqdm(
            top_terms, total=len(top_terms), desc=tqdm_desc
        ):
            top_docs = sorted(postinglist.items(), key=lambda kv: kv[1], reverse=True)[
                : self.level_2_limit
            ]
            for curr_doc_id, curr_solution_term_tf in top_docs:
                total_cnt += 1
                curr_student_tf = self.method_wrapper(
                    student_method,
                    inverted_index=student_inverted_index,
                    doc_id=curr_doc_id,
                    term=curr_term,
                )
                if curr_student_tf != curr_solution_term_tf:
                    mismatches.append(
                        (curr_term, curr_doc_id, curr_solution_term_tf, curr_student_tf)
                    )

                set_score(
                    round((total_cnt - len(mismatches)) * max_score / total_cnt, 2)
                )
                self.assertion_wrapper(
                    self.assertEqual,
                    len(mismatches),
                    0,
                    debug_msg=f"Term Frequency Mismatch Found! Mismatch Length: {len(mismatches)}\n"
                    + "\n".join(
                        [
                            f"Expect TF '{solution_term_tf}' but '{curr_student_tf}' received for Term: {term} at Doc: {doc_id}."
                            for term, doc_id, solution_term_tf, curr_student_tf in mismatches
                        ]
                    ),
                    show_debug_msg=show_debug_msg,
                )

    def df_tester(
        self,
        solution_inverted_index: InvertedIndex,
        student_inverted_index: InvertedIndex,
        tqdm_desc: str = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        student_method = self.client.ref("df")
        total_cnt = 0
        mismatches = []

        top_terms = sorted(
            solution_inverted_index.df_cache.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[: self.level_1_limit]

        for term, solution_term_df in tqdm(
            top_terms,
            total=len(top_terms),
            desc=tqdm_desc,
        ):
            total_cnt += 1
            curr_student_df = self.method_wrapper(
                student_method,
                inverted_index=student_inverted_index,
                term=term,
            )
            if curr_student_df != solution_term_df:
                mismatches.append((term, solution_term_df, curr_student_df))

        set_score(round((total_cnt - len(mismatches)) * max_score / total_cnt, 2))
        self.assertion_wrapper(
            self.assertEqual,
            len(mismatches),
            0,
            debug_msg=f"Document Frequency Mismatch Found! Mismatch Length: {len(mismatches)}\n"
            + "\n".join(
                [
                    f"Expect DF '{solution_term_df}' but '{curr_student_df}' received for Term: {term}."
                    for term, solution_term_df, curr_student_df in mismatches
                ]
            ),
            show_debug_msg=show_debug_msg,
        )

    def cf_tester(
        self,
        solution_inverted_index: InvertedIndex,
        student_inverted_index: InvertedIndex,
        tqdm_desc: str = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> None:
        student_method = self.client.ref("cf")
        total_cnt = 0
        mismatches = []

        top_terms = sorted(
            solution_inverted_index.cf_cache.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[: self.level_1_limit]
        for term, solution_term_cf in tqdm(
            top_terms,
            total=len(top_terms),
            desc=tqdm_desc,
        ):
            total_cnt += 1
            curr_student_cf = self.method_wrapper(
                student_method,
                inverted_index=student_inverted_index,
                term=term,
            )
            if curr_student_cf != solution_term_cf:
                mismatches.append((term, solution_term_cf, curr_student_cf))

        set_score(round((total_cnt - len(mismatches)) * max_score / total_cnt, 2))
        self.assertion_wrapper(
            self.assertEqual,
            len(mismatches),
            0,
            debug_msg=f"Collection Frequency Mismatch Found! Mismatch Length: {len(mismatches)}\n"
            + "\n".join(
                [
                    f"Expect CF '{solution_term_cf}' but '{curr_student_cf}' received for Term: {term}."
                    for term, solution_term_cf, curr_student_cf in mismatches
                ]
            ),
            show_debug_msg=show_debug_msg,
        )

    def term_statistics_tester(
        self,
        tag_name: str,
        metric_func: Callable,
        *metric_args,
        tqdm_desc: str = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
        **metric_kwargs,
    ):
        try:
            self.checker(show_debug_msg=show_debug_msg)
            stored_inverted_index = self.prerequisite_check(
                prerequisite_fn_tags=["parse_inverted_index"],
                prerequisite_fn_names=[
                    "build_inverted_index",
                ],
                show_debug_msg=show_debug_msg,
            )

            solution_inverted_index: InvertedIndex = stored_inverted_index[
                "parse_inverted_index"
            ]["solution"]
            student_inverted_index: InvertedIndex = stored_inverted_index[
                "parse_inverted_index"
            ]["student"]

            metric_name = metric_func.__name__
            if metric_name == "tf":
                tester = self.tf_tester
            elif metric_name == "df":
                tester = self.df_tester
            elif metric_name == "cf":
                tester = self.cf_tester
            else:
                raise ValueError(f"Invalid Metric Name: {metric_name}")

            tester(
                solution_inverted_index=solution_inverted_index,
                student_inverted_index=student_inverted_index,
                tqdm_desc=tqdm_desc,
                set_score=set_score,
                max_score=max_score,
                show_debug_msg=show_debug_msg,
            )

            self.save_class_attr(
                tag_name=tag_name,
                results_set=["Pass", "Pass"],
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertion_wrapper(
                self.assertTrue,
                False,
                debug_msg=f"Code does not compile:\n{e}",
                show_debug_msg=show_debug_msg,
            )

    def retrieval_model_tester(
        self,
        tag_name: str,
        prerequisite_fn_tags: list[str],
        retrieval_model_func: Callable,
        *retrieval_model_args,
        tqdm_desc: str = None,
        set_score: Callable = None,
        max_score: int = None,
        show_debug_msg: DebugMsgConfig = None,
        **retrieval_model_kwargs,
    ):
        try:
            metric_allowance_threshold = 1.5e-4
            self.checker(show_debug_msg=show_debug_msg)

            # make sure all previous individual metric tests are passed
            stored_inverted_index = self.prerequisite_check(
                prerequisite_fn_tags=["parse_inverted_index"] + prerequisite_fn_tags,
                prerequisite_fn_names=["build_inverted_index", "tf", "df", "cf"],
                show_debug_msg=show_debug_msg,
            )

            solution_inverted_index: InvertedIndex = stored_inverted_index[
                "parse_inverted_index"
            ]["solution"]
            student_inverted_index: InvertedIndex = stored_inverted_index[
                "parse_inverted_index"
            ]["student"]

            student_func = self.client.ref(retrieval_model_func.__name__)

            solution_results: dict[str, list[tuple[str, float]]] = retrieval_model_func(
                inverted_index=solution_inverted_index,
                queries=self.queries,
                *retrieval_model_args,
                **retrieval_model_kwargs,
            )

            student_results: dict[str, list[tuple[str, float]]] = student_func(
                inverted_index=student_inverted_index,
                queries=self.queries,
                *retrieval_model_args,
                **retrieval_model_kwargs,
            )

            self.assertion_wrapper(
                self.assertEqual,
                len(solution_results),
                len(student_results),
                debug_msg="The number of queries in the results ranklists does not match.\n"
                + f"Expect '{len(solution_results)}' queries' output but '{len(student_results)}' received!\n",
                show_debug_msg=show_debug_msg,
            )

            query_not_found = []
            partial_score_0_8 = [] # ranklist is correct while score is incorrect
            partial_score_0_5 = [] # ranklist is incorrect while score is correct
            fully_correct = []

            for query_id, solution_query_ranklist in tqdm(
                solution_results.items(), total=len(solution_results), desc=tqdm_desc
            ):
                if query_id not in student_results:
                    query_not_found.append(query_id)
                    continue
                student_query_ranklist = student_results[query_id]

                solution_docs = [doc for doc, _ in solution_query_ranklist]
                student_docs = [doc for doc, _ in student_query_ranklist]

                solution_scores = [score for _, score in solution_query_ranklist]
                student_scores = [score for _, score in student_query_ranklist]

                # Check if document lists are identical
                if solution_docs == student_docs:
                    # Check if all corresponding scores are within the acceptable threshold
                    if all(abs(sol_score - stu_score) <= metric_allowance_threshold for sol_score, stu_score in zip(solution_scores, student_scores)):
                        fully_correct.append(query_id)
                    else:
                        partial_score_0_8.append((query_id, solution_query_ranklist, student_query_ranklist))
                else:
                    # Check if any scores match within the threshold
                    score_match = any(abs(sol_score - stu_score) <= metric_allowance_threshold for sol_score, stu_score in zip(solution_scores, student_scores))
                    if score_match:
                        partial_score_0_5.append(query_id)

            debug_msg = ""
            if query_not_found:
                debug_msg += f"\nQuery not Found in the Results! Missing Queries: {query_not_found}\n"
            
            if partial_score_0_8:
                debug_msg += f"\nQueries with Correct Documents but Incorrect Scores: {[query for query, _, _ in partial_score_0_8]}\n"

            if partial_score_0_5:
                debug_msg += f"\nQueries with Incorrect Documents but Some Correct Scores: {[query for query, _, _ in partial_score_0_5]}\n"

            ranklist_mismatch = partial_score_0_8 + partial_score_0_5
            if ranklist_mismatch:
                debug_msg += f"\nIncorrect Query Results:\n"
                +"\n".join(
                    [
                        f"Query: {query_id}\n\tExpected Ranklist: {solution_query_ranklist}\n\tReceived Ranklist: {student_query_ranklist}"
                        for query_id, solution_query_ranklist, student_query_ranklist in ranklist_mismatch
                    ]
                )

            total_cnt = len(solution_results)
            total_score = (
                len(fully_correct) * 1.0 +
                len(partial_score_0_8) * 0.8 +
                len(partial_score_0_5) * 0.5
            )
            set_score(round((total_score / total_cnt) * max_score, 2))

            self.assertion_wrapper(
                self.assertEqual,
                len(debug_msg),
                0,
                debug_msg=debug_msg,
                show_debug_msg=show_debug_msg,
            )

            self.save_class_attr(
                tag_name=tag_name,
                results_set=["Pass", "Pass"],
            )
        except AssertionError:
            raise
        except Exception as e:
            self.assertion_wrapper(
                self.assertTrue,
                False,
                debug_msg=f"Code does not compile:\n{e}",
                show_debug_msg=show_debug_msg,
            )
