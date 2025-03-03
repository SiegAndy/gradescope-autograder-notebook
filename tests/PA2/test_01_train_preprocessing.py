from gradescope_utils.autograder_utils.decorators import (
    number,
    visibility,
    weight,
    partial_credit,
)

from tests.PA2.solution import (
    stemming_s,
    stopping,
    tokenize_4grams,
    tokenize_fancy,
    tokenize_space,
)
from tests.PA2.base import TestPA2, allowed_imports


class TestNotebookCompilable(TestPA2):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            test_type="public",
            allowed_imports=allowed_imports,
        )

    @weight(0)
    @visibility("visible")
    @number("0.1")
    def test_0_01_ipynb_compilable_and_packages(self):
        """Checking Notebook Integrity and Packages"""
        self.checker(show_debug_msg=None)

    @partial_credit(0)
    @visibility("visible")
    @number("1.1")
    def test_1_01_sample_tokenize_space(self, set_score=None):
        """Checking tokenize_space()"""
        self.no_prerequisite_tester(
            solution_function=tokenize_space,
            tag_name="sample_tokenize_space",
            tqdm_desc="test_sample_tokenize_space",
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.2")
    def test_1_02_sample_tokenize_4grams(self, set_score=None):
        """Checking tokenize_4grams()"""
        self.no_prerequisite_tester(
            solution_function=tokenize_4grams,
            tag_name="sample_tokenize_4grams",
            tqdm_desc="test_sample_tokenize_4grams",
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.3")
    def test_1_03_sample_tokenize_fancy(self, set_score=None):
        """Checking tokenize_fancy()"""
        self.prerequisite_tester(
            solution_function=tokenize_fancy,
            tag_name="sample_tokenize_fancy",
            tqdm_desc="test_sample_tokenize_fancy",
            prerequisite_fn_tags=["sample_tokenize_space"],
            prerequisite_fn_names=["tokenize_space"],
            use_prev_results=False,
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.4")
    def test_1_04_sample_tokenize_space_yesStopping(self, set_score=None):
        """Checking tokenize_space(), stopping()"""
        self.prerequisite_tester(
            solution_function=stopping,
            stopwords=self.stopwords,
            tag_name="sample_tokenize_space_yesStopping",
            tqdm_desc="test_sample_tokenize_space_yesStopping",
            prerequisite_fn_tags=["sample_tokenize_space"],
            prerequisite_fn_names=["tokenize_space"],
            use_prev_results=True,
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.5")
    def test_1_05_sample_tokenize_fancy_yesStopping(self, set_score=None):
        """Checking tokenize_fancy(), stopping()"""
        self.prerequisite_tester(
            solution_function=stopping,
            stopwords=self.stopwords,
            tag_name="sample_tokenize_fancy_yesStopping",
            tqdm_desc="test_sample_tokenize_fancy_yesStopping",
            prerequisite_fn_tags=["sample_tokenize_fancy"],
            prerequisite_fn_names=["tokenize_space", "tokenize_space"],
            use_prev_results=True,
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.6")
    def test_1_06_sample_tokenize_space_noStopping_and_stemming_s(self, set_score=None):
        """Checking tokenize_space(), stemming_s()"""
        self.prerequisite_tester(
            solution_function=stemming_s,
            tag_name="sample_tokenize_space_noStopping_and_stemming_s",
            tqdm_desc="test_sample_tokenize_space_noStopping_and_stemming_s",
            prerequisite_fn_tags=["sample_tokenize_space"],
            prerequisite_fn_names=["tokenize_space"],
            use_prev_results=True,
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.7")
    def test_1_07_sample_tokenize_fancy_noStopping_and_stemming_s(self, set_score=None):
        """Checking tokenize_fancy(), stemming_s()"""
        self.prerequisite_tester(
            solution_function=stemming_s,
            tag_name="sample_tokenize_fancy_noStopping_and_stemming_s",
            tqdm_desc="test_sample_tokenize_fancy_noStopping_and_stemming_s",
            prerequisite_fn_tags=["sample_tokenize_fancy"],
            prerequisite_fn_names=["tokenize_space", "tokenize_fancy"],
            use_prev_results=True,
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.8")
    def test_1_08_sample_preprocessing_space_yesStopping_and_stemming_porter(
        self, set_score=None
    ):
        """Checking preprocessing(tokenize_type="space", stopwords=stopwords, stemming_type="suffix_s")"""
        self.preprocessing_tester(
            tokenizer_type="space",
            stopwords=self.stopwords,
            stemming_type="suffix_s",
            num_of_sample_per_batch=4,
            tag_name="sample_preprocessing_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_sample_preprocessing_space_full",
            prerequisite_fn_tags=[
                "sample_tokenize_4grams",
                "sample_tokenize_space_yesStopping",
                "sample_tokenize_fancy_yesStopping",
                "sample_tokenize_space_noStopping_and_stemming_s",
                "sample_tokenize_fancy_noStopping_and_stemming_s",
            ],
            prerequisite_fn_names=[
                "tokenize_space",
                "tokenize_4grams",
                "tokenize_fancy",
                "stopping",
                "stemming_s",
            ],
            set_score=set_score,
            max_score=0,
        )

    @partial_credit(0)
    @visibility("visible")
    @number("1.9")
    def test_1_09_sample_statistics(self, set_score=None):
        """Checking freq_stats() using results from 1.8)"""
        self.zipf_tester(
            tag_name="sample_freq_stats",
            tqdm_desc="test_sample_freq_stats",
            prerequisite_fn_tags=[
                "sample_preprocessing_space_yesStopping_and_stemming_porter"
            ],
            prerequisite_fn_names=[
                'preprocessing(tokenize_type="space", stopwords=stopwords, stemming_type="suffix_s")'
            ],
            set_score=set_score,
            max_score=0,
        )
