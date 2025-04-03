from gradescope_utils.autograder_utils.decorators import (
    number,
    visibility,
    weight,
    partial_credit,
)
from tests.PA3.base import TestPA3
from tests.PA3.solution import tf, cf, df, tfm, bm25, ql


class TestPublic(TestPA3):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            test_type="public",
            level_1_limit=60,
            level_2_limit=10,
            allowed_imports=None,
        )

    @weight(0)
    @visibility("visible")
    @number("0.1")
    def test_0_01_ipynb_compilable_and_packages(self):
        """Checking Notebook Integrity and Packages"""
        self.checker(show_debug_msg=None)

    @weight(0)
    @visibility("visible")
    @number("0.2")
    def test_0_02_sample_build_inverted_index(self):
        """Checking build_inverted_index()"""
        self.inverted_index_parsing_tester(show_debug_msg=None)

    @partial_credit(0)
    @visibility("visible")
    @number("1.1")
    def test_1_01_tf(self, set_score=None):
        """Checking tf()"""
        self.term_statistics_tester(
            tag_name="sample_tf",
            metric_func=tf,
            tqdm_desc="test_sample_tf",
            set_score=set_score,
            max_score=0,
            show_debug_msg=None,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.2")
    def test_1_02_df(self, set_score=None):
        """Checking df()"""
        self.term_statistics_tester(
            tag_name="sample_df",
            metric_func=df,
            tqdm_desc="test_sample_df",
            set_score=set_score,
            max_score=0,
            show_debug_msg=None,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.3")
    def test_1_03_cf(self, set_score=None):
        """Checking cf()"""
        self.term_statistics_tester(
            tag_name="sample_cf",
            metric_func=cf,
            tqdm_desc="test_sample_cf",
            set_score=set_score,
            max_score=0,
            show_debug_msg=None,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("2.1")
    def test_2_01_tf(self, set_score=None):
        """Checking tfm()"""
        self.retrieval_model_tester(
            tag_name="sample_tfm",
            prerequisite_fn_tags=["sample_tf", "sample_df", "sample_cf"],
            retrieval_model_func=tfm,
            top_k=50,
            tqdm_desc="test_sample_tfm",
            set_score=set_score,
            max_score=0,
            show_debug_msg=None,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("2.2")
    def test_2_02_tf(self, set_score=None):
        """Checking bm25()"""
        self.retrieval_model_tester(
            tag_name="sample_bm25",
            prerequisite_fn_tags=["sample_tf", "sample_df", "sample_cf"],
            retrieval_model_func=bm25,
            b=0.75,
            k1=1.2,
            top_k=50,
            tqdm_desc="test_sample_bm25",
            set_score=set_score,
            max_score=0,
            show_debug_msg=None,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("2.3")
    def test_2_03_tf(self, set_score=None):
        """Checking ql()"""
        self.retrieval_model_tester(
            tag_name="sample_ql",
            prerequisite_fn_tags=["sample_tf", "sample_df", "sample_cf"],
            retrieval_model_func=ql,
            lambda_factor=0.2,
            top_k=50,
            tqdm_desc="test_sample_ql",
            set_score=set_score,
            max_score=0,
            show_debug_msg=None,
        )
