from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Callable, List

from tests.base import (
    DebugMsgConfig,
    TestJupyterNotebook,
    tqdm,
)

from tests.PA1.solution import (
    Ranklist,
    autograder_version,
    evaluation,
    parse_query_relevance_mapping,
    parse_trecrun_results,
)


def function_name_to_metric_name(
    func: Callable | str, *func_args, **func_kwargs
) -> str:
    func_params = list(func_args)
    func_params.extend(func_kwargs.values())

    if isinstance(func, Callable):
        func_name = func.__name__

    if func_name == "reciprocal_rank":
        return f"RR@{func_params[0]}"
    elif func_name == "precision":
        return f"P@{func_params[0]}"
    elif func_name == "recall":
        return f"R@{func_params[0]}"
    elif func_name == "f1":
        return f"F1@{func_params[0]}"
    elif func_name == "average_precision":
        return f"AP@{func_params[0]}"
    elif func_name == "ndcg":
        return f"nDCG@{func_params[0]}"
    elif func_name == "binary_preference":
        return "BPREF"
    elif func_name == "interpolated_precision":
        return f"P@{func_params[0]}%R"
    elif func_name == "r_precision":
        return f"P@R"
    else:
        raise ValueError(f"Unexpected function input: {func.__name__}")


class TestPA1(TestJupyterNotebook):
    qrels_filepath: str  # path to qrels file
    trecrun_filepaths: dict[str, str]  # mapping from model_name to trecrun filepath

    query_to_qrels_mapping: dict[
        str, dict[str, int]
    ]  # mapping from query_id to [doc_id to doc_rel_score]
    solution_ranklists_dict: dict[
        str, dict[str, Ranklist]
    ]  # mapping from metric name to [query_id to query ranklist]

    allowed_imports: List[str]

    @classmethod
    def setUpClass(
        cls,
        data_folder_path: str,
        allowed_imports: List[str] = None,
    ):
        super().setUpClass(autograder_version=autograder_version)
        cls.allowed_imports = allowed_imports

        cls.qrels_filepath = Path(os.path.join(data_folder_path, "msmarco.qrels"))
        trecrun_filepaths = list(Path(data_folder_path).glob("*.trecrun"))

        cls.trecrun_filepaths = dict()
        for filepath in trecrun_filepaths:
            if "bm25" in filepath.name:
                cls.trecrun_filepaths["bm25"] = filepath
            elif "ql" in filepath.name:
                cls.trecrun_filepaths["ql"] = filepath
            elif "dpr" in filepath.name:
                cls.trecrun_filepaths["dpr"] = filepath

        cls.query_to_qrels_mapping = parse_query_relevance_mapping(cls.qrels_filepath)
        cls.solution_ranklists_dict = {
            model_name: parse_trecrun_results(
                trecrun_filepath, cls.query_to_qrels_mapping
            )
            for model_name, trecrun_filepath in cls.trecrun_filepaths.items()
        }

    def prerequisite_check(
        self,
        prerequisite_test_tags: list[str] = None,
        prerequisite_func_names: list[str] = None,
        show_debug_msg: DebugMsgConfig = None,
    ) -> tuple[Any, Any]:
        """
        Check whether expected class variable is stored and is not None.

        If not, it means the previous test is failed which fails all downstream tasks.

        Example attr_name: "tokenize_fancy_{store_type}"
        """
        try:
            if len(prerequisite_test_tags) == 0 or len(prerequisite_test_tags) == 0:
                return

            error_msg = f"Need to first pass the test(s) for function(s): {', '.join(prerequisite_func_names)} !"

            result_attrs = defaultdict(dict)
            for test_tag in prerequisite_test_tags:
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
                # print("=" * 50)
                # print(show_debug_msg.test_tag)
                # print("=" * 50)
                # print(e)
                # print("=" * 50)
                raise
        else:
            assertion_method(*assertion_params, debug_msg)

    def trecrun_parsing_tester(
        self,
        tqdm_desc: str = None,
        show_debug_msg: DebugMsgConfig = None,
    ):

        try:
            tag_name = "parse_files"
            self.checker(show_debug_msg=show_debug_msg)

            # get target functions' reflection
            # student_parse_query_relevance_mapping_func = self.client.ref(
            #     "parse_query_relevance_mapping"
            # )
            # student_parse_trecrun_results_func = self.client.ref(
            #     "parse_trecrun_results"
            # )

            # self.clear_notebook_output(50)
            # student_per_query_doc_rel_mapping = self.method_wrapper(
            #     student_parse_query_relevance_mapping_func,
            #     qrels_filepath=str(self.qrels_filepath),
            # )

            student_results = dict()

            self.client.inject(
                f"""
                query_doc_rel_mapping = parse_query_relevance_mapping(r"{self.qrels_filepath}")
                """,
                pop=True,
            )

            for model_name, trecrun_filepath in tqdm(
                self.trecrun_filepaths.items(), desc=tqdm_desc
            ):
                solution_model_ranklists = self.solution_ranklists_dict[model_name]

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(50)

                self.client.inject(
                    f"""
                        {model_name}_trecrun_ranklists = parse_trecrun_results(
                            r"{trecrun_filepath}", query_doc_rel_mapping
                        )
                    """,
                    pop=True,
                )

                # check the output of notebook
                # student_ranklists_results: dict[str, Ranklist] = self.method_wrapper(
                #     student_parse_trecrun_results_func,
                #     trecrun_filepath=str(trecrun_filepath),
                #     query_doc_rel_mapping=student_per_query_doc_rel_mapping,
                # )

                student_ranklists_results: dict[str, Ranklist] = self.client.ref(
                    f"{model_name}_trecrun_ranklists"
                )

                self.assertion_wrapper(
                    self.assertEqual,
                    len(solution_model_ranklists.keys()),
                    len(student_ranklists_results.keys()),
                    debug_msg="Output length does not match.\n"
                    + f"Only '{len(student_ranklists_results.keys())}' ranklists parsed for '{len(solution_model_ranklists.keys())}' queries.\n"
                    + f"Expects queries:\n\t[{', '.join([str(key) for key in sorted(list(student_ranklists_results.keys()), key=lambda x: int(x))])}].\n"
                    + f"but received: [{', '.join([str(key) for key in sorted(list(student_ranklists_results.keys()), key=lambda x: int(x))])}]\n",
                    show_debug_msg=show_debug_msg,
                )

                student_results[model_name] = student_ranklists_results

            self.save_class_attr(
                tag_name=tag_name,
                results_set=[self.solution_ranklists_dict, student_results],
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

    def individual_evaluation_metric_tester(
        self,
        tag_name: str,
        test_trecrun_model_types: list[str],  # bm25, ql, and/or dpr
        metric_func: Callable,
        *metric_args,
        tqdm_desc: str = None,
        show_debug_msg: DebugMsgConfig = None,
        **metric_kwargs,
    ):
        metric_allowance_threshold = 1.5e-4
        try:
            self.checker(show_debug_msg=show_debug_msg)
            stored_ranklists_results = self.prerequisite_check(
                prerequisite_test_tags=["parse_files"],
                prerequisite_func_names=[
                    "parse_query_relevance_mapping",
                    "parse_trecrun_results",
                ],
                show_debug_msg=show_debug_msg,
            )

            solution_ranklists_results = stored_ranklists_results["parse_files"][
                "solution"
            ]
            # student_ranklists_results = stored_ranklists_results["parse_files"][
            #     "student"
            # ]

            metric_name = metric_func.__name__
            # get target function reflection
            # student_evaluation_metric_func = self.client.ref(metric_name)

            all_model_solution_results, all_model_student_results = dict(), dict()
            # print(solution_ranklists_results)
            # print(student_ranklists_results)
            for curr_test_model_name in tqdm(
                test_trecrun_model_types,
                desc=tqdm_desc,
            ):
                solution_results, student_results = dict(), dict()
                for query_id in solution_ranklists_results[curr_test_model_name].keys():

                    solution_ranklist = solution_ranklists_results[
                        curr_test_model_name
                    ][query_id]
                    # student_ranklist = student_ranklists_results[curr_test_model_name][
                    #     query_id
                    # ]
                    # compute using solution code
                    curr_solution_result = metric_func(
                        solution_ranklist, *metric_args, **metric_kwargs
                    )
                    solution_results[query_id] = curr_solution_result

                    # cleanup the notebook output as ipython core will give out warning
                    # when output has 200? more line and corrupts reflection of the function output
                    self.clear_notebook_output(50)

                    self.client.inject(
                        f"""
                        curr_result = {metric_name}(
                            {curr_test_model_name}_trecrun_ranklists["{query_id}"],
                            {", ".join(metric_args) + ", " if len(metric_args) > 0 else ""}
                            {", ".join(f"{key}={value}" for key, value in metric_kwargs.items()) if len(metric_kwargs.keys()) > 0 else ""}
                            )
                        """,
                        pop=True,
                    )

                    curr_solution_result = self.client.ref("curr_result")
                    # print(f"curr_result: {curr_solution_result}")
                    # check the output of notebook
                    # student_results = self.method_wrapper(
                    #     student_evaluation_metric_func,
                    #     student_ranklist,
                    #     *metric_args,
                    #     **metric_kwargs,
                    # )
                    student_results[query_id] = curr_solution_result

                # print(student_results)
                query_ids_mismatched_results = []
                for query_id in solution_ranklists_results[curr_test_model_name].keys():
                    if (
                        abs(solution_results[query_id] - student_results[query_id])
                        > metric_allowance_threshold
                    ):
                        query_ids_mismatched_results.append(query_id)

                query_ids_mismatched_results = sorted(
                    query_ids_mismatched_results, key=lambda x: int(x)
                )

                metric_printout_name = function_name_to_metric_name(
                    metric_func, *metric_args, **metric_kwargs
                )
                self.assertion_wrapper(
                    self.assertEqual,
                    len(query_ids_mismatched_results),
                    0,
                    debug_msg=f"\nIncorrect {metric_printout_name} ('{curr_test_model_name}') results for queries: [{', '.join(query_ids_mismatched_results)}]\n"
                    + f"Expected values: [{', '.join([str(solution_results[query_id]) for query_id in query_ids_mismatched_results])}],\n"
                    + f"but received: [{', '.join([str(student_results[query_id]) for query_id in query_ids_mismatched_results])}]!\n",
                    show_debug_msg=show_debug_msg,
                )
                all_model_solution_results[curr_test_model_name] = solution_results
                all_model_student_results[curr_test_model_name] = student_results

            self.save_class_attr(
                tag_name=tag_name,
                results_set=[all_model_solution_results, all_model_student_results],
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

    def evaluation_func_tester(
        self,
        tag_name: str,
        test_trecrun_model_types: list[str],  # bm25, ql, and/or dpr
        evaluation_params_dict: dict[Callable, list],
        prerequisite_test_tags: list[
            str
        ],  # a list of individual evaluation tester tags, where each corresponding to a metric which is specified in evaluation_params_dict
        tqdm_desc: str = None,
        show_debug_msg: DebugMsgConfig = None,
    ):
        try:
            metric_allowance_threshold = 1.5e-4
            self.checker(show_debug_msg=show_debug_msg)

            required_func_names = [
                func.__name__ for func in list(evaluation_params_dict.keys())
            ]

            # make sure all previous individual metric tests are passed
            self.prerequisite_check(
                prerequisite_test_tags=prerequisite_test_tags,
                prerequisite_func_names=required_func_names,
                show_debug_msg=show_debug_msg,
            )

            # get target function reflection
            # student_evaluation_func = self.client.ref("evaluation")
            student_evaluation_params_dict_str = (
                "{"
                + ", ".join(
                    [
                        f"""{func.__name__}: [{
                            ", ".join([
                                str(param) if not isinstance(param, str) else f'"{param}"' for param in curr_evaluation_params 
                            ])
                        }]"""
                        for func, curr_evaluation_params in evaluation_params_dict.items()
                    ]
                )
                + "}"
            )

            all_model_solution_results, all_model_student_results = dict(), dict()

            for curr_test_model_name in tqdm(
                test_trecrun_model_types,
                desc=tqdm_desc,
            ):
                curr_model_trecrun_filepath = self.trecrun_filepaths[
                    curr_test_model_name
                ]

                solution_evaluation_results = evaluation(
                    trecrun_filepath=str(curr_model_trecrun_filepath),
                    qrels_filepath=str(self.qrels_filepath),
                    evaluation_params=evaluation_params_dict,
                )

                solution_evaluation_results_dict = defaultdict(dict)
                for curr_entry in solution_evaluation_results:
                    curr_metric_name, curr_query_id, curr_metric_value = curr_entry
                    solution_evaluation_results_dict[curr_metric_name.lower()][
                        curr_query_id
                    ] = curr_metric_value

                # cleanup the notebook output as ipython core will give out warning
                # when output has 200? more line and corrupts reflection of the function output
                self.clear_notebook_output(50)

                # check the output of notebook
                # student_evaluation_results = self.method_wrapper(
                #     student_evaluation_func,
                #     trecrun_filepath=str(curr_model_trecrun_filepath),
                #     qrels_filepath=str(self.qrels_filepath),
                #     evaluation_params=student_evaluation_params_dict,
                # )
                self.client.inject(
                    f"""
                    student_evaluation_results = evaluation(
                        trecrun_filepath=r"{curr_model_trecrun_filepath}",
                        qrels_filepath=r"{self.qrels_filepath}",
                        evaluation_params={student_evaluation_params_dict_str},
                        )
                    """,
                    pop=True,
                )

                student_evaluation_results = self.client.ref(
                    "student_evaluation_results"
                )

                self.assertion_wrapper(
                    self.assertEqual,
                    len(solution_evaluation_results),
                    len(student_evaluation_results),
                    debug_msg=f"Output Length Mismatched!\n"
                    + f"Expected {len(solution_evaluation_results)} evalution entries but {len(student_evaluation_results)} received!",
                    show_debug_msg=show_debug_msg,
                )

                student_evaluation_results_dict = defaultdict(dict)
                for curr_entry in student_evaluation_results:
                    self.assertion_wrapper(
                        self.assertEqual,
                        len(curr_entry),
                        3,
                        debug_msg=f"\nIncorrect Output Format!\n"
                        + f"Expected format for each evalution entry is: (metric,queryid,score), but only '{len(curr_entry)}' elements received.",
                        show_debug_msg=show_debug_msg,
                    )
                    curr_metric_name, curr_query_id, curr_metric_value = curr_entry
                    curr_metric_name = curr_metric_name.lower()

                    self.assertion_wrapper(
                        self.assertIn,
                        curr_metric_name,
                        solution_evaluation_results_dict,
                        debug_msg=f"\nUnknown Output Metric: '{curr_metric_name}'!\n"
                        + f"Expected output metrics are: [{', '.join(list(solution_evaluation_results_dict.keys()))}],\n"
                        + f"but '{curr_metric_name}' received.",
                        show_debug_msg=show_debug_msg,
                    )

                    self.assertion_wrapper(
                        self.assertIn,
                        curr_query_id,
                        solution_evaluation_results_dict[curr_metric_name],
                        debug_msg=f"\nUnknown Output Query ID: '{curr_query_id}'!\n"
                        + f"Expected output query ids are: [{', '.join(list(solution_evaluation_results_dict[curr_metric_name].keys()))}],\n"
                        + f"but '{curr_query_id}' received.",
                        show_debug_msg=show_debug_msg,
                    )

                    if (
                        abs(
                            solution_evaluation_results_dict[curr_metric_name][
                                curr_query_id
                            ]
                            - curr_metric_value
                        )
                        > metric_allowance_threshold
                    ):
                        self.assertion_wrapper(
                            self.assertEqual,
                            curr_metric_value,
                            solution_evaluation_results_dict[curr_metric_name][
                                curr_query_id
                            ],
                            debug_msg=f"\nIncorrect Value for Metric: '{curr_metric_name}', Query ID: '{curr_query_id}', Value: '{curr_metric_value}'!\n"
                            + f"Expected output is: '{str(solution_evaluation_results_dict[curr_metric_name][curr_query_id])}', but '{curr_metric_value}' received.",
                            show_debug_msg=show_debug_msg,
                        )

                    student_evaluation_results_dict[curr_metric_name][
                        curr_query_id
                    ] = curr_metric_value

                all_model_solution_results[curr_test_model_name] = (
                    solution_evaluation_results_dict
                )
                all_model_student_results[curr_test_model_name] = (
                    student_evaluation_results_dict
                )

            self.save_class_attr(
                tag_name=tag_name,
                results_set=[all_model_solution_results, all_model_student_results],
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
