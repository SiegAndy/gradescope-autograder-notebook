from gradescope_utils.autograder_utils.decorators import (
    number,
    visibility,
    weight,
    hide_errors,
)

from tests.PA1.solution import (
    ndcg,
    reciprocal_rank,
    precision,
    recall,
    f1,
    average_precision,
    binary_preference,
    interpolated_precision,
    r_precision,
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
    @number("0.3")
    def test_0_03_parse_qrels_trecrun(self):
        """Checking parse_trecrun_results()"""
        self.trecrun_parsing_tester(
            tqdm_desc="test_parse_qrels_trecrun",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(3.1) Checking parse_trecrun_results()",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.1")
    def test_2_01_reciprocal_rank(self):
        """Checking reciprocal_rank() @ 50"""
        self.individual_evaluation_metric_tester(
            tag_name="reciprocal_rank_at_50",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=reciprocal_rank,
            tqdm_desc="test_reciprocal_rank_at_50",
            top_k=50,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.1) Checking reciprocal_rank() @ 50",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.2")
    def test_2_02_precision(self):
        """Checking precision() @ 23"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_23",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision,
            top_k=23,
            tqdm_desc="test_precision_at_23",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.2) Checking precision() @ 23",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.3")
    def test_2_03_precision(self):
        """Checking precision() @ 50"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_50",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision,
            top_k=50,
            tqdm_desc="test_precision_at_50",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.3) Checking precision() @ 50",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.4")
    def test_2_04_recall(self):
        """Checking recall() @ 17"""
        self.individual_evaluation_metric_tester(
            tag_name="recall_at_17",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=recall,
            top_k=17,
            tqdm_desc="test_recall_at_17",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.4) Checking recall() @ 17",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.5")
    def test_2_05_recall(self):
        """Checking recall() @ 60"""
        self.individual_evaluation_metric_tester(
            tag_name="recall_at_60",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=recall,
            top_k=60,
            tqdm_desc="test_recall_at_60",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.5) Checking recall() @ 60",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.6")
    def test_2_06_f1(self):
        """Checking f1() @ 29"""
        self.individual_evaluation_metric_tester(
            tag_name="f1_at_29",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=f1,
            top_k=29,
            tqdm_desc="test_f1_at_29",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.6) Checking f1() @ 29",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.7")
    def test_2_07_f1(self):
        """Checking f1() @ 1000"""
        self.individual_evaluation_metric_tester(
            tag_name="f1_at_1000",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=f1,
            top_k=1000,
            tqdm_desc="test_f1_at_1000",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.7) Checking f1() @ 1000",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.8")
    def test_2_08_average_precision(self):
        """Checking average_precision() @ 46"""
        self.individual_evaluation_metric_tester(
            tag_name="average_precision_at_46",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=46,
            tqdm_desc="test_average_precision_at_46",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.8) Checking average_precision() @ 46",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.9")
    def test_2_09_average_precision(self):
        """Checking average_precision() @ 25"""
        self.individual_evaluation_metric_tester(
            tag_name="average_precision_at_25",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=25,
            tqdm_desc="test_average_precision_at_25",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.9) Checking average_precision() @ 25",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.10")
    def test_2_10_ndcg(self):
        """Checking ndcg() @ 22"""
        self.individual_evaluation_metric_tester(
            tag_name="ndcg_at_22",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=ndcg,
            top_k=22,
            tqdm_desc="test_ndcg_at_22",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.10) Checking ndcg() @ 22",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.11")
    def test_2_11_ndcg(self):
        """Checking ndcg() @ 500"""
        self.individual_evaluation_metric_tester(
            tag_name="ndcg_at_500",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=25,
            tqdm_desc="test_ndcg_at_500",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.11) Checking ndcg() @ 500",
            ),
        )

    @weight(12)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.12")
    def test_2_12_binary_preference(self):
        """Checking binary_preference()"""
        self.individual_evaluation_metric_tester(
            tag_name="binary_preference",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=binary_preference,
            tqdm_desc="test_binary_preference",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.12) Checking binary_preference()",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.13")
    def test_2_13_interpolated_precision(self):
        """Checking interpolated_precision() @ 13% Recall"""
        self.individual_evaluation_metric_tester(
            tag_name="interpolated_precision_at_13",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=interpolated_precision,
            recall_percentile=13,
            tqdm_desc="test_interpolated_precision_at_13",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.13) Checking interpolated_precision() @ 13% Recall",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.14")
    def test_2_14_r_precision(self):
        """Checking r_precision()"""
        self.individual_evaluation_metric_tester(
            tag_name="r_precision",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=r_precision,
            tqdm_desc="test_r_precision",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(2.14) Checking r_precision()",
            ),
        )

    @weight(10)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.1")
    def test_3_01_evaluation(self):
        """Checking evaluation()"""
        self.evaluation_func_tester(
            tag_name="evaluation",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            evaluation_params_dict={
                reciprocal_rank: [40],
                precision: [55],
                recall: [60],
                f1: [65],
                average_precision: [70],
                ndcg: [10],
                binary_preference: [],
                # interpolated_precision: [25],
                # r_precision: [],
            },
            prerequisite_test_tags=[
                "reciprocal_rank_at_50",
                "precision_at_23",
                "precision_at_50",
                "recall_at_17",
                "recall_at_60",
                "f1_at_29",
                "f1_at_1000",
                "average_precision_at_46",
                "average_precision_at_25",
                "ndcg_at_22",
                "ndcg_at_500",
                "binary_preference",
                # "interpolated_precision_at_19",
                # "interpolated_precision_at_67",
                # "r_precision",
            ],
            tqdm_desc="test_evaluation",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(3.1) Checking evaluation()",
            ),
        )

    @weight(0)
    @visibility("hidden")
    @number("0.0")
    def test_9_99_instructor_debug_msg(self):
        """Debug Messages From Previous Failed Private Tests"""
        self.assertFalse(
            hasattr(self.__class__, "hidden_debug_msg"),
            getattr(
                self.__class__, "hidden_debug_msg", "No hidden debug message found!"
            ),
        )
