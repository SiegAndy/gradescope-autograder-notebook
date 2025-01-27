from gradescope_utils.autograder_utils.decorators import (
    number,
    visibility,
    weight,
    hide_errors,
)

from tests.PA1.solution import (
    reciprocal_rank,
    precision,
    recall,
    f1,
    average_precision,
    binary_preference,
    precision_at_recall_precentile,
    precision_at_recall,
)

from tests.PA1.base import TestPA1
from tests.base import DebugMsgConfig


class TestNotebookCompilable(TestPA1):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            data_folder_path="./data/2025-Spring-P1/data/test/",
            allowed_imports=["re"],
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.1")
    def test_31_parse_qrels_trecrun(self):
        """Checking parse_trecrun_results()"""
        self.trecrun_parsing_tester(
            tqdm_desc="test_parse_qrels_trecrun",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(3.1) Checking parse_trecrun_results()",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.1")
    def test_41_reciprocal_rank(self):
        """Checking reciprocal_rank()"""
        self.individual_evaluation_metric_tester(
            tag_name="reciprocal_rank",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=reciprocal_rank,
            tqdm_desc="test_reciprocal_rank",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.1) Checking reciprocal_rank()",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.2")
    def test_42_precision(self):
        """Checking precision() @ 23"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_23",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision,
            top_k=23,
            tqdm_desc="test_precision_at_23",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.2) Checking precision() @ 23",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.3")
    def test_43_recall(self):
        """Checking recall() @ 17"""
        self.individual_evaluation_metric_tester(
            tag_name="recall_at_17",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=recall,
            top_k=17,
            tqdm_desc="test_recall_at_17",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.3) Checking recall() @ 17",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.4")
    def test_44_f1(self):
        """Checking f1() @ 29"""
        self.individual_evaluation_metric_tester(
            tag_name="f1_at_29",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=f1,
            top_k=29,
            tqdm_desc="test_f1_at_29",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.4) Checking f1() @ 29",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.5")
    def test_45_average_precision(self):
        """Checking average_precision() @ 46"""
        self.individual_evaluation_metric_tester(
            tag_name="average_precision_at_46",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=46,
            tqdm_desc="test_average_precision_at_46",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.5) Checking average_precision() @ 46",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.6")
    def test_46_binary_preference(self):
        """Checking binary_preference()"""
        self.individual_evaluation_metric_tester(
            tag_name="binary_preference",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=binary_preference,
            tqdm_desc="test_binary_preference",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.6) Checking binary_preference()",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.7")
    def test_47_precision_at_recall_precentile(self):
        """Checking precision_at_recall_precentile() @ 19% Recall"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_recall_precentile_at_19",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision_at_recall_precentile,
            recall_precentile=19,
            tqdm_desc="test_precision_at_recall_precentile_at_19",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.7) Checking precision_at_recall_precentile() @ 19% Recall",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.8")
    def test_48_precision_at_recall(self):
        """Checking precision_at_recall()"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_recall",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision_at_recall,
            tqdm_desc="test_precision_at_recall",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.8) Checking precision_at_recall()",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("5.1")
    def test_51_evaluation(self):
        """Checking evaluation()"""
        self.evaluation_func_tester(
            tag_name="evaluation",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            evaluation_params_dict={
                reciprocal_rank: [],
                precision: [55],
                recall: [60],
                f1: [65],
                average_precision: [70],
                binary_preference: [],
                precision_at_recall_precentile: [25],
                precision_at_recall: [],
            },
            prerequisite_test_tags=[
                "reciprocal_rank",
                "precision_at_23",
                "recall_at_17",
                "f1_at_29",
                "average_precision_at_46",
                "binary_preference",
                "precision_at_recall_precentile_at_19",
                "precision_at_recall",
            ],
            tqdm_desc="test_evaluation",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(5.1) Checking evaluation()",
            ),
        )

    @weight(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("5.2")
    def test_52_average_precision_comparison(self):
        """Checking comparison_average_precision_improvement()"""
        self.ap_comparison_func_tester(
            top_k=20,
            prerequisite_test_tags=[
                "evaluation",
            ],
            tqdm_desc="test_average_precision_comparison",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(5.2) Checking comparison_average_precision_improvement()",
            ),
        )

    @weight(0)
    @visibility("hidden")
    @number("0.0")
    def test_99_instructor_debug_msg(self):
        """Debug Messages From Previous Failed Private Tests"""
        self.assertFalse(
            hasattr(self.__class__, "hidden_debug_msg"),
            getattr(
                self.__class__, "hidden_debug_msg", "No hidden debug message found!"
            ),
        )
