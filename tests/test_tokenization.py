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


class TestTokenization(TestPA1):
    def setUp(self):
        return super().setUp()
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            data_file_path="webpage.gz",
            allowed_imports=["re"],
        )
        cls.sentences = cls.sentences[:500]

    @weight(5)
    @visibility("after_due_date")
    @number("2.1")
    def test_21_tokenize_space(self):
        """Checking tokenize_space()"""
        self.no_prerequisite_tester(
            function_name="tokenize_space",
            solution_function=tokenize_space,
            tag_name="tokenized_space",
            tqdm_desc="test_tokenize_space",
        )

    @weight(10)
    @visibility("after_due_date")
    @number("2.2")
    def test_22_tokenize_4grams(self):
        """Checking tokenize_4grams()"""
        self.no_prerequisite_tester(
            function_name="tokenize_4grams",
            solution_function=tokenize_4grams,
            tag_name="tokenized_4grams",
            tqdm_desc="test_tokenize_4grams",
        )

    @weight(5)
    @visibility("after_due_date")
    @number("2.3")
    def test_23_tokenize_fancy(self):
        """Checking tokenize_fancy()"""
        self.no_prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_fancy",
            tqdm_desc="test_tokenize_fancy",
        )

    @weight(5)
    @visibility("after_due_date")
    @number("2.4")
    def test_24_tokenize_space_fancy(self):
        """Checking tokenize_space(), tokenize_fancy()"""
        self.prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_space_fancy",
            tqdm_desc="test_tokenize_space_fancy",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
            input_is_list=True,
        )

    @weight(5)
    @visibility("after_due_date")
    @number("3.1")
    def test_31_tokenize_space_yesStopping(self):
        """Checking tokenize_space(), stopping()"""
        self.prerequisite_tester(
            function_name="stopping",
            solution_function=stopping,
            tag_name="tokenized_space_yesStopping",
            tqdm_desc="test_tokenize_space_stopping",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
            stopwords=self.stopwords,
        )

    @weight(5)
    @visibility("after_due_date")
    @number("3.2")
    def test_32_tokenize_space_fancy_yesStopping(self):
        """Checking tokenize_space(), tokenize_fancy(), stopping()"""
        self.prerequisite_tester(
            function_name="stopping",
            solution_function=stopping,
            tag_name="tokenized_space_fancy_yesStopping",
            tqdm_desc="test_tokenize_space_fancy_stopping",
            prerequisite=(
                "tokenized_space_fancy_{store_type}",
                "tokenize_space, tokenize_fancy",
            ),
            stopwords=self.stopwords,
            input_is_list=True,
        )

    @weight(5)
    @visibility("after_due_date")
    @number("4.1")
    def test_41_tokenize_space_noStopping_and_stemming_s(self):
        """Checking tokenize_space(), stemming_s()"""
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenized_space_noStopping_and_stemming_s",
            tqdm_desc="test_tokenize_space_noStopping_and_stemming_s",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(5)
    @visibility("after_due_date")
    @number("4.2")
    def test_42_tokenize_space_yesStopping_and_stemming_porter(self):
        """Checking tokenize_space(), stopping(), stemming_porter()"""
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenized_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_space_yesStopping_and_stemming_porter",
            prerequisite=(
                "tokenized_space_yesStopping_{store_type}",
                "tokenize_space, stopping",
            ),
        )

    @weight(5)
    @visibility("after_due_date")
    @number("4.3")
    def test_43_tokenize_space_fancy_yesStopping_and_stemming_s(self):
        """Checking tokenize_space(), tokenize_fancy(), stopping(), stemming_s()"""
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenize_space_fancy_yesStopping_and_stemming_s",
            tqdm_desc="test_tokenize_space_fancy_yesStopping_and_stemming_s",
            prerequisite=(
                "tokenized_space_fancy_yesStopping_{store_type}",
                "tokenize_space, tokenize_fancy, stopping",
            ),
            input_is_list=True,
        )

    @weight(5)
    @visibility("after_due_date")
    @number("4.4")
    def test_44_tokenize_space_fancy_noStopping_and_stemming_porter(self):
        """Checking tokenize_space(), tokenize_fancy(), stemming_porter()"""
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_space_fancy_noStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_space_fancy_noStopping_and_stemming_porter",
            prerequisite=(
                "tokenized_space_fancy_{store_type}",
                "tokenize_space, tokenize_fancy",
            ),
            input_is_list=True,
        )

    @weight(5)
    @visibility("after_due_date")
    @number("4.5")
    def test_45_tokenize_space_fancy_yesStopping_and_stemming_porter(self):
        """Checking tokenize_space(), tokenize_fancy(), stopping(), stemming_porter()"""
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_space_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_space_fancy_yesStopping_and_stemming_porter",
            prerequisite=(
                "tokenized_space_fancy_yesStopping_{store_type}",
                "tokenize_space, tokenize_fancy, stopping",
            ),
            input_is_list=True,
        )

    @weight(10)
    @visibility("after_due_date")
    @number("5.1")
    def test_51_tokenization_space_yesStopping_and_stemming_porter(self):
        """Checking tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")"""
        self.tokenization_tester(
            tag_name="tokenization_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenization_space_full",
            stopwords=self.stopwords,
            tokenizer_type="space",
            stemming_type="porter",
        )

    @weight(10)
    @visibility("after_due_date")
    @number("5.2")
    def test_52_tokenization_fancy_yesStopping_and_stemming_porter(self):
        """Checking tokenization(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")"""
        self.tokenization_tester(
            tag_name="tokenization_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenization_fancy_full",
            stopwords=self.stopwords,
            tokenizer_type="fancy",
            stemming_type="porter",
        )

    @weight(4)
    @visibility("after_due_date")
    @number("6.1")
    def test_61_heaps_tokenization_space_yesStopping_and_stemming_porter(self):
        """Checking heaps() using results from 5.1"""
        self.heaps_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(4)
    @visibility("after_due_date")
    @number("6.2")
    def test_61_heaps_tokenization_fancy_yesStopping_and_stemming_porter(self):
        """Checking heaps() using results from 5.2"""
        self.heaps_tester(
            prerequisite=(
                "tokenization_fancy_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(2.5)
    @visibility("after_due_date")
    @number("7.1")
    def test_71_statistics_tokenization_space_yesStopping_and_stemming_porter(self):
        """Checking statistics() using results from 5.1"""
        self.zipf_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(2.5)
    @visibility("after_due_date")
    @number("7.2")
    def test_72_statistics_tokenization_fancy_yesStopping_and_stemming_porter(self):
        """Checking statistics() using results from 5.2"""
        self.zipf_tester(
            prerequisite=(
                "tokenization_fancy_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")',
            )
        )
