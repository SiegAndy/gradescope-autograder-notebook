from gradescope_utils.autograder_utils.decorators import (
    number,
    visibility,
    weight,
    partial_credit,
    hide_errors,
)
from tests.base import DebugMsgConfig
from tests.PA3.base import TestPA3
from tests.PA3.solution import tf, cf, df, tfm, bm25, ql


class TestProtected(TestPA3):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            test_type="protected",
            level_1_limit=100,
            level_2_limit=10,
            allowed_imports=None,
        )

    @weight(0)
    @visibility("visible")
    @number("0.3")
    def test_0_03_ipynb_compilable_and_packages(self):
        """Checking Notebook Integrity and Packages"""
        self.checker(
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(0.3) Checking Notebook Integrity and Packages",
            ),
        )

    @weight(0)
    @visibility("visible")
    @number("0.4")
    def test_0_04_sample_parse_qrels_trecrun(self):
        """Checking build_inverted_index()"""
        self.inverted_index_parsing_tester(
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(0.4) Checking build_inverted_index()",
            ),
        )

    @partial_credit(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.1")
    def test_3_01_tf(self, set_score=None):
        """Checking tf()"""
        self.term_statistics_tester(
            tag_name="tf",
            metric_func=tf,
            tqdm_desc="test_tf",
            set_score=set_score,
            max_score=0,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(3.1) Checking tf()",
            ),
        )

    @partial_credit(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.2")
    def test_3_02_df(self, set_score=None):
        """Checking df()"""
        self.term_statistics_tester(
            tag_name="df",
            metric_func=df,
            tqdm_desc="test_df",
            set_score=set_score,
            max_score=0,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(3.2) Checking df()",
            ),
        )

    @partial_credit(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.3")
    def test_3_03_cf(self, set_score=None):
        """Checking cf()"""
        self.term_statistics_tester(
            tag_name="cf",
            metric_func=cf,
            tqdm_desc="test_cf",
            set_score=set_score,
            max_score=0,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(3.3) Checking cf()",
            ),
        )

    @partial_credit(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.1")
    def test_4_01_tf(self, set_score=None):
        """Checking tfm()"""
        self.retrieval_model_tester(
            tag_name="tfm",
            prerequisite_fn_tags=["tf", "df", "cf"],
            retrieval_model_func=tfm,
            top_k=50,
            tqdm_desc="test_tfm",
            set_score=set_score,
            max_score=0,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.1) Checking tfm()",
            ),
        )

    @partial_credit(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.2")
    def test_4_02_tf(self, set_score=None):
        """Checking bm25()"""
        self.retrieval_model_tester(
            tag_name="bm25",
            prerequisite_fn_tags=["tf", "df", "cf"],
            retrieval_model_func=bm25,
            b=0.75,
            k1=1.2,
            top_k=50,
            tqdm_desc="test_bm25",
            set_score=set_score,
            max_score=0,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.2) Checking bm25()",
            ),
        )

    @partial_credit(0)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.3")
    def test_4_03_tf(self, set_score=None):
        """Checking ql()"""
        self.retrieval_model_tester(
            tag_name="ql",
            prerequisite_fn_tags=["tf", "df", "cf"],
            retrieval_model_func=ql,
            lambda_factor=0.2,
            top_k=50,
            tqdm_desc="test_ql",
            set_score=set_score,
            max_score=0,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(4.3) Checking ql()",
            ),
        )

    @weight(0)
    @visibility("after_published")
    @number("0.0")
    def test_9_99_instructor_debug_msg(self):
        """Debug Messages From Previous Failed Private Tests"""
        self.assertFalse(
            hasattr(self.__class__, "hidden_debug_msg"),
            getattr(
                self.__class__, "hidden_debug_msg", "No hidden debug message found!"
            ),
        )
