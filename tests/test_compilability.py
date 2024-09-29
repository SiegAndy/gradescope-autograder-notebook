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
    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            data_file_path="P1-train.gz",
            allowed_imports=["re"],
        )

    @weight(0)
    @visibility("visible")
    @number("0.1.1")
    def test_11_ipynb_compilable_and_packages(self):
        self.checker()

    @weight(0)
    @visibility("visible")
    @number("0.2.1")
    def test_21_sample_tokenize_space(self):
        self.no_prerequisite_tester(
            function_name="tokenize_space",
            solution_function=tokenize_space,
            tag_name="tokenized_space",
            tqdm_desc="test_sample_tokenize_space",
        )

    @weight(0)
    @visibility("visible")
    @number("0.2.2")
    def test_22_sample_tokenize_4grams(self):
        self.no_prerequisite_tester(
            function_name="tokenize_4grams",
            solution_function=tokenize_4grams,
            tag_name="tokenized_4grams",
            tqdm_desc="test_sample_tokenize_4grams",
        )

    @weight(0)
    @visibility("visible")
    @number("0.2.3")
    def test_23_sample_tokenize_fancy(self):
        self.no_prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_fancy",
            tqdm_desc="test_sample_tokenize_fancy",
        )

    @weight(0)
    @visibility("visible")
    @number("0.3.1")
    def test_31_sample_tokenize_fancy_yesStopping(self):
        self.prerequisite_tester(
            function_name="stopping",
            solution_function=stopping,
            tag_name="tokenized_fancy_yesStopping",
            tqdm_desc="test_sample_stopping",
            prerequisite=("tokenized_fancy_{store_type}", "tokenize_fancy"),
            stopwords=self.stopwords,
        )

    @weight(0)
    @visibility("visible")
    @number("0.4.1")
    def test_41_sample_tokenize_space_noStopping_and_stemming_s(self):
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenized_space_noStopping_and_stemming_s",
            tqdm_desc="test_sample_tokenize_space_noStopping_and_stemming_s",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(0)
    @visibility("visible")
    @number("0.4.2")
    def test_42_sample_tokenize_space_noStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenized_space_noStopping_and_stemming_porter",
            tqdm_desc="test_sample_tokenize_space_noStopping_and_stemming_porter",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(0)
    @visibility("visible")
    @number("0.4.3")
    def test_43_sample_tokenize_fancy_yesStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_sample_tokenize_fancy_yesStopping_and_stemming_porter",
            prerequisite=(
                "tokenized_fancy_yesStopping_{store_type}",
                "tokenize_fancy, stopping",
            ),
        )

    @weight(0)
    @visibility("visible")
    @number("0.5.1")
    def test_51_sample_tokenization_space_yesStopping_and_stemming_porter(self):
        self.tokenization_tester(
            tag_name="tokenization_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_sample_tokenization_space_full",
            stopwords=self.stopwords,
            tokenizer_type="space",
            stemming_type="porter",
        )

    @weight(0)
    @visibility("visible")
    @number("0.6.1")
    def test_61_sample_heaps(self):
        self.heaps_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(0)
    @visibility("visible")
    @number("0.7.1")
    def test_71_sample_statistics(self):
        self.zipf_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )
