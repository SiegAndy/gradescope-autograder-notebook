from gradescope_utils.autograder_utils.decorators import (
    hide_errors,
    number,
    visibility,
    partial_credit,
)

from tests.PA2.solution import (
    stemming_s,
    stopping,
    tokenize_4grams,
    tokenize_fancy,
    tokenize_space,
)
from tests.PA2.base import DebugMsgConfig, TestPA2, allowed_imports


class Testpreprocessing(TestPA2):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            test_type="protected",
            allowed_imports=allowed_imports,
        )
        cls.sentences = cls.sentences[:500]

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.1")
    def test_2_01_tokenize_space(self, set_score=None):
        """Checking tokenize_space()"""
        self.no_prerequisite_tester(
            solution_function=tokenize_space,
            tag_name="tokenize_space",
            tqdm_desc="test_tokenize_space",
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(2.1) Checking tokenize_space()""",
            ),
        )

    @partial_credit(10)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.2")
    def test_2_02_tokenize_4grams(self, set_score=None):
        """Checking tokenize_4grams()"""
        self.no_prerequisite_tester(
            solution_function=tokenize_4grams,
            tag_name="tokenize_4grams",
            tqdm_desc="test_tokenize_4grams",
            set_score=set_score,
            max_score=10,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(2.2) Checking tokenize_4grams()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("2.3")
    def test_2_03_tokenize_fancy(self, set_score=None):
        """Checking tokenize_fancy()"""
        self.prerequisite_tester(
            solution_function=tokenize_fancy,
            tag_name="tokenize_fancy",
            tqdm_desc="test_tokenize_fancy",
            prerequisite_fn_tags=["tokenize_space"],
            prerequisite_fn_names=["tokenize_space"],
            use_prev_results=False,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(2.3) Checking tokenize_fancy()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.1")
    def test_3_01_tokenize_space_yesStopping(self, set_score=None):
        """Checking tokenize_space(), stopping()"""
        self.prerequisite_tester(
            solution_function=stopping,
            stopwords=self.stopwords,
            tag_name="tokenize_space_yesStopping",
            tqdm_desc="test_tokenize_space_stopping",
            prerequisite_fn_tags=["tokenize_space"],
            prerequisite_fn_names=["tokenize_space"],
            use_prev_results=True,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(3.1) Checking tokenize_space(), stopping()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("3.2")
    def test_3_02_tokenize_space_fancy_yesStopping(self, set_score=None):
        """Checking tokenize_fancy(), stopping()"""
        self.prerequisite_tester(
            solution_function=stopping,
            stopwords=self.stopwords,
            tag_name="tokenize_fancy_yesStopping",
            tqdm_desc="test_tokenize_space_fancy_stopping",
            prerequisite_fn_tags=["tokenize_fancy"],
            prerequisite_fn_names=["tokenize_space", "tokenize_fancy"],
            use_prev_results=True,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(3.2) Checking tokenize_fancy(), stopping()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.1")
    def test_4_01_tokenize_space_noStopping_and_stemming_s(self, set_score=None):
        """Checking tokenize_space(), stemming_s()"""
        self.prerequisite_tester(
            solution_function=stemming_s,
            tag_name="tokenize_space_noStopping_and_stemming_s",
            tqdm_desc="test_tokenize_space_noStopping_and_stemming_s",
            prerequisite_fn_tags=["tokenize_space"],
            prerequisite_fn_names=["tokenize_space"],
            use_prev_results=True,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(4.1) Checking tokenize_space(), stemming_s()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.2")
    def test_4_02_tokenize_space_yesStopping_and_stemming_s(self, set_score=None):
        """Checking tokenize_space(), stopping(), stemming_s()"""
        self.prerequisite_tester(
            solution_function=stemming_s,
            tag_name="tokenize_space_yesStopping_and_stemming_s",
            tqdm_desc="test_tokenize_space_yesStopping_and_stemming_s",
            prerequisite_fn_tags=["tokenize_space_yesStopping"],
            prerequisite_fn_names=["tokenize_space", "stopping"],
            use_prev_results=True,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(4.2) Checking tokenize_space(), stopping(), stemming_s()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.3")
    def test_4_03_tokenize_space_fancy_noStopping_and_stemming_s(self, set_score=None):
        """Checking tokenize_fancy(), stemming_s()"""
        self.prerequisite_tester(
            solution_function=stemming_s,
            tag_name="tokenize_fancy_noStopping_and_stemming_s",
            tqdm_desc="test_tokenize_fancy_noStopping_and_stemming_s",
            prerequisite_fn_tags=["tokenize_fancy"],
            prerequisite_fn_names=["tokenize_space", "tokenize_fancy"],
            use_prev_results=True,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(4.3) Checking tokenize_fancy(), stemming_s()""",
            ),
        )

    @partial_credit(5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("4.4")
    def test_4_04_tokenize_space_fancy_yesStopping_and_stemming_s(self, set_score=None):
        """Checking tokenize_fancy(), stopping(), stemming_s()"""
        self.prerequisite_tester(
            solution_function=stemming_s,
            tag_name="tokenize_fancy_yesStopping_and_stemming_s",
            tqdm_desc="test_tokenize_fancy_yesStopping_and_stemming_s",
            prerequisite_fn_tags=["tokenize_fancy_yesStopping"],
            prerequisite_fn_names=["tokenize_space", "tokenize_fancy", "stopping"],
            use_prev_results=True,
            set_score=set_score,
            max_score=5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(4.4) Checking tokenize_fancy(), stopping(), stemming_s()""",
            ),
        )

    @partial_credit(10)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("5.1")
    def test_5_01_preprocessing_space_yesStopping_and_stemming_porter(
        self, set_score=None
    ):
        """Checking preprocessing(tokenize_type="space", stopwords=stopwords, stemming_type="porter")"""
        self.preprocessing_tester(
            tokenizer_type="space",
            stopwords=self.stopwords,
            stemming_type="porter",
            num_of_sample_per_batch=4,
            tag_name="preprocessing_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_preprocessing_space_full",
            prerequisite_fn_tags=[
                "tokenize_4grams",
                "tokenize_space_noStopping_and_stemming_s",
                "tokenize_space_yesStopping_and_stemming_s",
                "tokenize_fancy_noStopping_and_stemming_s",
                "tokenize_fancy_yesStopping_and_stemming_s",
            ],
            prerequisite_fn_names=[
                "tokenize_space",
                "tokenize_4grams",
                "tokenize_fancy",
                "stopping",
                "stemming_s",
            ],
            set_score=set_score,
            max_score=10,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(5.1) Checking preprocessing(tokenize_type="space", stopwords=stopwords, stemming_type="porter")""",
            ),
        )

    @partial_credit(10)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("5.2")
    def test_5_02_preprocessing_fancy_yesStopping_and_stemming_porter(
        self, set_score=None
    ):
        """Checking preprocessing(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")"""
        self.preprocessing_tester(
            tokenizer_type="fancy",
            stopwords=self.stopwords,
            stemming_type="porter",
            num_of_sample_per_batch=4,
            tag_name="preprocessing_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_preprocessing_space_full",
            prerequisite_fn_tags=[
                "tokenize_4grams",
                "tokenize_space_noStopping_and_stemming_s",
                "tokenize_space_yesStopping_and_stemming_s",
                "tokenize_fancy_noStopping_and_stemming_s",
                "tokenize_fancy_yesStopping_and_stemming_s",
            ],
            prerequisite_fn_names=[
                "tokenize_space",
                "tokenize_4grams",
                "tokenize_fancy",
                "stopping",
                "stemming_s",
            ],
            set_score=set_score,
            max_score=10,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="""(5.2) Checking preprocessing(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")""",
            ),
        )

    @partial_credit(2.5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("6.1")
    def test_6_01_statistics_preprocessing_space_yesStopping_and_stemming_porter(
        self, set_score=None
    ):
        """Checking freq_stats() using results from 5.1)"""
        self.zipf_tester(
            tag_name="freq_stats_preprocessing_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_freq_stats_preprocessing_space_yesStopping_and_stemming_porter",
            prerequisite_fn_tags=[
                "preprocessing_space_yesStopping_and_stemming_porter"
            ],
            prerequisite_fn_names=[
                'preprocessing(tokenize_type="space", stopwords=stopwords, stemming_type="porter")'
            ],
            set_score=set_score,
            max_score=2.5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(6.1) Checking freq_stats() using results from 5.1)",
            ),
        )

    @partial_credit(2.5)
    @visibility("visible")
    @hide_errors("Test failed!")
    @number("6.2")
    def test_6_02_statistics_preprocessing_fancy_yesStopping_and_stemming_porter(
        self, set_score=None
    ):
        """Checking freq_stats() using results from 5.2)"""
        self.zipf_tester(
            tag_name="freq_stats_preprocessing_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_freq_stats_preprocessing_fancy_yesStopping_and_stemming_porter",
            prerequisite_fn_tags=[
                "preprocessing_fancy_yesStopping_and_stemming_porter"
            ],
            prerequisite_fn_names=[
                'preprocessing(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")'
            ],
            set_score=set_score,
            max_score=2.5,
            show_debug_msg=DebugMsgConfig(
                show_msg_in_orig_test=False,
                test_tag="(6.2) Checking statistics() using results from 5.2)",
            ),
        )

    @partial_credit(0)
    @visibility("after_published")
    @number("0.0")
    def test_9_99_instructor_debug_msg(self, set_score=None):
        """Debug Messages From Previous Failed Private Tests"""
        self.assertFalse(
            hasattr(self.__class__, "hidden_debug_msg"),
            getattr(
                self.__class__, "hidden_debug_msg", "No hidden debug message found!"
            ),
        )
