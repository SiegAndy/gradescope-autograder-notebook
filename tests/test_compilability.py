import unittest

from gradescope_utils.autograder_utils.decorators import number, visibility, weight

from tests import (
    stemming_porter,
    stemming_s,
    stopping,
    tokenize_4grams,
    tokenize_fancy,
    tokenize_space,
)
from tests.pa1 import TestPA1


class TestNotebookCompilable(TestPA1):
    def setUp(self):
        return super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            data_file_path="P1-train.gz",
            allowed_imports=["re"],
        )

    @weight(0)
    @visibility("visible")
    @number("1.1")
    def test_1_ipynb_compilable_and_packages(self):
        """Checking Notebook Integrity and Packages"""
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("1.2")
    def test_2_sample_tokenize_space(self):
        """Checking tokenize_space()"""
        self.no_prerequisite_tester(
            function_name="tokenize_space",
            solution_function=tokenize_space,
            tag_name="tokenized_space",
            tqdm_desc="test_sample_tokenize_space",
        )

    @weight(0)
    @visibility("visible")
    @number("1.3")
    def test_3_sample_tokenize_4grams(self):
        """Checking tokenize_4grams()"""
        self.no_prerequisite_tester(
            function_name="tokenize_4grams",
            solution_function=tokenize_4grams,
            tag_name="tokenized_4grams",
            tqdm_desc="test_sample_tokenize_4grams",
        )

    @weight(0)
    @visibility("visible")
    @number("1.4")
    def test_4_sample_tokenize_fancy(self):
        """Checking tokenize_fancy()"""
        self.no_prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_fancy",
            tqdm_desc="test_sample_tokenize_fancy",
        )

    @weight(0)
    @visibility("visible")
    @number("1.5")
    def test_5_sample_tokenize_space_and_fancy(self):
        """Checking tokenize_space(), tokenize_fancy()"""
        self.prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_space_fancy",
            tqdm_desc="test_sample_tokenize_space_fancy",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
            input_is_list=True,
        )

    @weight(0)
    @visibility("visible")
    @number("1.6")
    def test_6_sample_tokenize_space_fancy_yesStopping(self):
        """Checking tokenize_space(), tokenize_fancy(), stopping()"""
        self.prerequisite_tester(
            function_name="stopping",
            solution_function=stopping,
            tag_name="tokenized_space_fancy_yesStopping",
            tqdm_desc="test_sample_space_fancy_stopping",
            prerequisite=(
                "tokenized_space_fancy_{store_type}",
                "tokenize_space, tokenize_fancy",
            ),
            stopwords=self.stopwords,
            input_is_list=True,
        )

    @weight(0)
    @visibility("visible")
    @number("1.7")
    def test_7_sample_tokenize_space_noStopping_and_stemming_s(self):
        """Checking tokenize_space(), stemming_s()"""
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenized_space_noStopping_and_stemming_s",
            tqdm_desc="test_sample_tokenize_space_noStopping_and_stemming_s",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(0)
    @visibility("visible")
    @number("1.8")
    def test_8_sample_tokenize_space_noStopping_and_stemming_porter(self):
        """Checking tokenize_space(), stemming_porter()"""
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenized_space_noStopping_and_stemming_porter",
            tqdm_desc="test_sample_tokenize_space_noStopping_and_stemming_porter",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(0)
    @visibility("visible")
    @number("1.9")
    def test_9_sample_tokenize_space_fancy_yesStopping_and_stemming_porter(self):
        """Checking tokenize_space(), tokenize_fancy(), stopping(), stemming_porter()"""
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_space_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_sample_tokenize_space_fancy_yesStopping_and_stemming_porter",
            prerequisite=(
                "tokenized_space_fancy_yesStopping_{store_type}",
                "tokenize_space, tokenize_fancy, stopping",
            ),
            input_is_list=True,
        )

    @weight(0)
    @visibility("visible")
    @number("1.10")
    def test_10_sample_tokenization_space_yesStopping_and_stemming_porter(self):
        """Checking tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")"""
        self.tokenization_tester(
            tag_name="tokenization_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_sample_tokenization_space_full",
            stopwords=self.stopwords,
            tokenizer_type="space",
            stemming_type="porter",
        )

    @weight(0)
    @visibility("visible")
    @number("1.11")
    def test_11_sample_heaps(self):
        """Checking heaps() using 1.10 results"""
        self.heaps_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(0)
    @visibility("visible")
    @number("1.12")
    def test_12_sample_statistics(self):
        """Checking statistics() using 1.10 results"""
        self.zipf_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )
