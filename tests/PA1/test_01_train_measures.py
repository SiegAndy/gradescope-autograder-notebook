from gradescope_utils.autograder_utils.decorators import number, visibility, weight

from tests.PA1.solution import (
    ndcg,
    reciprocal_rank,
    precision,
    recall,
    f1,
    average_precision,
    binary_preference,
    precision_at_recall_percentile,
    precision_at_recall,
)

from tests.PA1.base import TestPA1


class TestNotebookCompilable(TestPA1):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            data_folder_path="./data/2025-Spring-P1/data/train/",
            allowed_imports=["re"],
        )

    @weight(0)
    @visibility("visible")
    @number("0.1")
    def test_0_1_ipynb_compilable_and_packages(self):
        """Checking Notebook Integrity and Packages"""
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("0.2")
    def test_0_2_sample_parse_qrels_trecrun(self):
        """Checking parse_trecrun_results()"""
        self.trecrun_parsing_tester(
            tqdm_desc="test_sample_parse_qrels_trecrun",
        )

    @weight(0)
    @visibility("visible")
    @number("1.1")
    def test_1_1_sample_reciprocal_rank(self):
        """Checking reciprocal_rank() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_reciprocal_rank_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=reciprocal_rank,
            top_k=10,
            tqdm_desc="test_sample_reciprocal_rank_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.2")
    def test_1_2_sample_precision(self):
        """Checking precision() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_precision_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision,
            top_k=10,
            tqdm_desc="test_sample_precision_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.3")
    def test_1_3_sample_recall(self):
        """Checking recall() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_recall_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=recall,
            top_k=10,
            tqdm_desc="test_sample_recall_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.4")
    def test_1_4_sample_f1(self):
        """Checking f1()"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_f1_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=f1,
            top_k=10,
            tqdm_desc="test_sample_f1_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.5")
    def test_1_5_sample_average_precision(self):
        """Checking average_precision() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_average_precision_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=average_precision,
            top_k=10,
            tqdm_desc="test_sample_average_precision_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.6")
    def test_1_6_ndcg(self):
        """Checking ndcg() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="ndcg_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=ndcg,
            top_k=10,
            tqdm_desc="test_sample_ndcg_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.7")
    def test_1_7_sample_binary_preference(self):
        """Checking binary_preference()"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_binary_preference",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=binary_preference,
            tqdm_desc="test_sample_binary_preference",
        )

    @weight(0)
    @visibility("visible")
    @number("1.8")
    def test_1_8_sample_precision_at_recall_percentile(self):
        """Checking precision_at_recall_percentile() @ 5% Recall"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_precision_at_recall_percentile_at_5",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision_at_recall_percentile,
            recall_percentile=10,
            tqdm_desc="test_sample_precision_at_recall_percentile_at_5",
        )

    @weight(0)
    @visibility("visible")
    @number("1.9")
    def test_1_9_sample_precision_at_recall(self):
        """Checking precision_at_recall()"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_precision_at_recall",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=precision_at_recall,
            tqdm_desc="test_sample_precision_at_recall",
        )

    @weight(0)
    @visibility("visible")
    @number("2.1")
    def test_2_1_sample_evaluation(self):
        """Checking evaluation()"""
        self.evaluation_func_tester(
            tag_name="sample_evaluation",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            evaluation_params_dict={
                reciprocal_rank: [10],
                precision: [10],
                recall: [10],
                f1: [10],
                average_precision: [10],
                ndcg: [10],
                binary_preference: [],
                precision_at_recall_percentile: [5],
                precision_at_recall: [],
            },
            prerequisite_test_tags=[
                "sample_reciprocal_rank_at_10",
                "sample_precision_at_10",
                "sample_recall_at_10",
                "sample_f1_at_10",
                "sample_average_precision_at_10",
                "sample_ndcg_at_10",
                "sample_binary_preference",
                # "sample_precision_at_recall_percentile_at_5",
                # "sample_precision_at_recall",
            ],
            tqdm_desc="test_sample_evaluation",
        )

    @weight(0)
    @visibility("visible")
    @number("2.2")
    def test_2_2_sample_average_precision_comparison(self):
        """Checking comparison_average_precision_improvement()"""
        self.ap_comparison_func_tester(
            top_k=10,
            prerequisite_test_tags=[
                "sample_evaluation",
            ],
            tqdm_desc="test_sample_average_precision_comparison",
        )
