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

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.1")
    def test_4_01_reciprocal_rank(self):
        """Checking reciprocal_rank() @ 50"""
        self.individual_evaluation_metric_tester(
            tag_name="reciprocal_rank_at_50",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=reciprocal_rank,
            tqdm_desc="test_reciprocal_rank_at_50",
            top_k=50,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.1) Checking reciprocal_rank() @ 50",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.2")
    def test_4_02_precision(self):
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

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.3")
    def test_4_03_precision(self):
        """Checking precision() @ 50"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_50",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision,
            top_k=50,
            tqdm_desc="test_precision_at_50",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.3) Checking precision() @ 50",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.4")
    def test_4_04_recall(self):
        """Checking recall() @ 17"""
        self.individual_evaluation_metric_tester(
            tag_name="recall_at_17",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=recall,
            top_k=17,
            tqdm_desc="test_recall_at_17",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.4) Checking recall() @ 17",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.5")
    def test_4_05_recall(self):
        """Checking recall() @ 60"""
        self.individual_evaluation_metric_tester(
            tag_name="recall_at_60",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=recall,
            top_k=60,
            tqdm_desc="test_recall_at_60",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.5) Checking recall() @ 60",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.6")
    def test_4_06_f1(self):
        """Checking f1() @ 29"""
        self.individual_evaluation_metric_tester(
            tag_name="f1_at_29",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=f1,
            top_k=29,
            tqdm_desc="test_f1_at_29",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.6) Checking f1() @ 29",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.7")
    def test_4_07_f1(self):
        """Checking f1() @ 1000"""
        self.individual_evaluation_metric_tester(
            tag_name="f1_at_1000",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=f1,
            top_k=1000,
            tqdm_desc="test_f1_at_1000",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.7) Checking f1() @ 1000",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.8")
    def test_4_08_average_precision(self):
        """Checking average_precision() @ 46"""
        self.individual_evaluation_metric_tester(
            tag_name="average_precision_at_46",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=46,
            tqdm_desc="test_average_precision_at_46",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.8) Checking average_precision() @ 46",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.9")
    def test_4_09_average_precision(self):
        """Checking average_precision() @ 25"""
        self.individual_evaluation_metric_tester(
            tag_name="average_precision_at_25",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=25,
            tqdm_desc="test_average_precision_at_25",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.9) Checking average_precision() @ 25",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.10")
    def test_4_10_ndcg(self):
        """Checking ndcg() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="ndcg_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=ndcg,
            top_k=10,
            tqdm_desc="test_ndcg_at_10",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.10) Checking ndcg() @ 10",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.11")
    def test_4_11_ndcg(self):
        """Checking ndcg() @ 500"""
        self.individual_evaluation_metric_tester(
            tag_name="ndcg_at_500",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=25,
            tqdm_desc="test_ndcg_at_500",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.11) Checking ndcg() @ 500",
            ),
        )

    @weight(12)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.12")
    def test_4_12_binary_preference(self):
        """Checking binary_preference()"""
        self.individual_evaluation_metric_tester(
            tag_name="binary_preference",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=binary_preference,
            tqdm_desc="test_binary_preference",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.12) Checking binary_preference()",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.13")
    def test_4_13_precision_at_recall_precentile(self):
        """Checking precision_at_recall_precentile() @ 19% Recall"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_recall_precentile_at_19",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision_at_recall_precentile,
            recall_precentile=19,
            tqdm_desc="test_precision_at_recall_precentile_at_19",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.13) Checking precision_at_recall_precentile() @ 19% Recall",
            ),
        )

    @weight(3)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.14")
    def test_4_14_precision_at_recall(self):
        """Checking precision_at_recall()"""
        self.individual_evaluation_metric_tester(
            tag_name="precision_at_recall",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision_at_recall,
            tqdm_desc="test_precision_at_recall",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.14) Checking precision_at_recall()",
            ),
        )

    @weight(10)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("5.1")
    def test_5_01_evaluation(self):
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
                # precision_at_recall_precentile: [25],
                # precision_at_recall: [],
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
                "ndcg_at_25",
                "ndcg_at_500",
                "binary_preference",
                # "precision_at_recall_precentile_at_19",
                # "precision_at_recall_precentile_at_67",
                # "precision_at_recall",
            ],
            tqdm_desc="test_evaluation",
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(5.1) Checking evaluation()",
            ),
        )

    @weight(6)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("5.2")
    def test_5_02_average_precision_comparison(self):
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
    def test_9_99_instructor_debug_msg(self):
        """Debug Messages From Previous Failed Private Tests"""
        self.assertFalse(
            hasattr(self.__class__, "hidden_debug_msg"),
            getattr(
                self.__class__, "hidden_debug_msg", "No hidden debug message found!"
            ),
        )
