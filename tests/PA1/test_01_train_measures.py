from gradescope_utils.autograder_utils.decorators import number, visibility, weight

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
    def test_0_01_ipynb_compilable_and_packages(self):
        """Checking Notebook Integrity and Packages"""
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("0.2")
    def test_0_02_sample_parse_qrels_trecrun(self):
        """Checking parse_trecrun_results()"""
        self.trecrun_parsing_tester(
            tqdm_desc="test_sample_parse_qrels_trecrun",
        )

    @weight(0)
    @visibility("visible")
    @number("1.1")
    def test_1_01_sample_reciprocal_rank(self):
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
    def test_1_02_sample_precision(self):
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
    def test_1_03_sample_recall(self):
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
    def test_1_04_sample_f1(self):
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
    def test_1_05_sample_average_precision(self):
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
    def test_1_06_sample_ndcg(self):
        """Checking ndcg() @ 10"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_ndcg_at_10",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=ndcg,
            top_k=10,
            tqdm_desc="test_sample_ndcg_at_10",
        )

    @weight(0)
    @visibility("visible")
    @number("1.7")
    def test_1_07_sample_binary_preference(self):
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
    def test_1_08_sample_interpolated_precision(self):
        """Checking interpolated_precision() @ 5% Recall"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_interpolated_precision_at_5",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=interpolated_precision,
            recall_percentile=10,
            tqdm_desc="test_sample_interpolated_precision_at_5",
        )

    @weight(0)
    @visibility("visible")
    @number("1.9")
    def test_1_09_sample_r_precision(self):
        """Checking r_precision()"""
        self.individual_evaluation_metric_tester(
            tag_name="sample_r_precision",
            test_trecrun_model_types=["bm25", "ql", "dpr"],
            metric_func=r_precision,
            tqdm_desc="test_sample_r_precision",
        )

    @weight(0)
    @visibility("visible")
    @number("1.10")
    def test_1_10_sample_evaluation(self):
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
                interpolated_precision: [5],
                r_precision: [],
            },
            prerequisite_test_tags=[
                "sample_reciprocal_rank_at_10",
                "sample_precision_at_10",
                "sample_recall_at_10",
                "sample_f1_at_10",
                "sample_average_precision_at_10",
                "sample_ndcg_at_10",
                "sample_binary_preference",
                # "sample_interpolated_precision_at_5",
                # "sample_r_precision",
            ],
            tqdm_desc="test_sample_evaluation",
        )
