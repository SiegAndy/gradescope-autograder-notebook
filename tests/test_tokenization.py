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
    @classmethod
    def setUpClass(cls):
        super().setUpClass(
            data_file_path="webpage.gz",
            allowed_imports=["re"],
        )
        cls.sentences = cls.sentences[:1000]

    @weight(5)
    @visibility("after_due_date")
    @number("1.1.1")
    def test_111_tokenize_space(self):
        self.no_prerequisite_tester(
            function_name="tokenize_space",
            solution_function=tokenize_space,
            tag_name="tokenized_space",
            tqdm_desc="test_tokenize_space",
        )

    @weight(10)
    @visibility("after_due_date")
    @number("1.1.2")
    def test_112_tokenize_4grams(self):
        self.no_prerequisite_tester(
            function_name="tokenize_4grams",
            solution_function=tokenize_4grams,
            tag_name="tokenized_4grams",
            tqdm_desc="test_tokenize_4grams",
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.1.3")
    def test_113_tokenize_fancy(self):
        self.no_prerequisite_tester(
            function_name="tokenize_fancy",
            solution_function=tokenize_fancy,
            tag_name="tokenized_fancy",
            tqdm_desc="test_tokenize_fancy",
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.2.1")
    def test_121_tokenize_space_yesStopping(self):
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
    @number("1.2.2")
    def test_122_tokenize_fancy_yesStopping(self):
        self.prerequisite_tester(
            function_name="stopping",
            solution_function=stopping,
            tag_name="tokenized_fancy_yesStopping",
            tqdm_desc="test_tokenize_fancy_stopping",
            prerequisite=("tokenized_fancy_{store_type}", "tokenize_fancy"),
            stopwords=self.stopwords,
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.3.1")
    def test_131_tokenize_space_noStopping_and_stemming_s(self):
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenized_space_noStopping_and_stemming_s",
            tqdm_desc="test_tokenize_space_noStopping_and_stemming_s",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.3.2")
    def test_132_tokenize_space_noStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenized_space_noStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_space_noStopping_and_stemming_porter",
            prerequisite=("tokenized_space_{store_type}", "tokenize_space"),
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.3.3")
    def test_133_tokenize_space_yesStopping_and_stemming_porter(self):
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
    @number("1.3.4")
    def test_134_tokenize_fancy_yesStopping_and_stemming_s(self):
        self.prerequisite_tester(
            function_name="stemming_s",
            solution_function=stemming_s,
            tag_name="tokenize_fancy_yesStopping_and_stemming_s",
            tqdm_desc="test_tokenize_fancy_yesStopping_and_stemming_s",
            prerequisite=(
                "tokenized_fancy_yesStopping_{store_type}",
                "tokenize_fancy, stopping",
            ),
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.3.5")
    def test_135_tokenize_fancy_noStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_fancy_noStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_fancy_noStopping_and_stemming_porter",
            prerequisite=("tokenized_fancy_{store_type}", "tokenize_fancy"),
        )

    @weight(5)
    @visibility("after_due_date")
    @number("1.3.6")
    def test_136_tokenize_fancy_yesStopping_and_stemming_porter(self):
        self.prerequisite_tester(
            function_name="stemming_porter",
            solution_function=stemming_porter,
            tag_name="tokenize_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenize_fancy_yesStopping_and_stemming_porter",
            prerequisite=(
                "tokenized_fancy_yesStopping_{store_type}",
                "tokenize_fancy, stopping",
            ),
        )

    @weight(10)
    @visibility("after_due_date")
    @number("1.4.1")
    def test_141_tokenization_space_yesStopping_and_stemming_porter(self):
        self.tokenization_tester(
            tag_name="tokenization_space_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenization_space_full",
            stopwords=self.stopwords,
            tokenizer_type="space",
            stemming_type="porter",
        )

    @weight(10)
    @visibility("after_due_date")
    @number("1.4.2")
    def test_142_tokenization_fancy_yesStopping_and_stemming_porter(self):
        self.tokenization_tester(
            tag_name="tokenization_fancy_yesStopping_and_stemming_porter",
            tqdm_desc="test_tokenization_fancy_full",
            stopwords=self.stopwords,
            tokenizer_type="fancy",
            stemming_type="porter",
        )

    @weight(4)
    @visibility("after_due_date")
    @number("1.5.1")
    def test_151_heaps_tokenization_space_yesStopping_and_stemming_porter(self):
        self.heaps_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(4)
    @visibility("after_due_date")
    @number("1.5.2")
    def test_151_heaps_tokenization_fancy_yesStopping_and_stemming_porter(self):
        self.heaps_tester(
            prerequisite=(
                "tokenization_fancy_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(2.5)
    @visibility("after_due_date")
    @number("1.6.1")
    def test_161_statistics_tokenization_space_yesStopping_and_stemming_porter(self):
        self.zipf_tester(
            prerequisite=(
                "tokenization_space_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="space", stopwords=stopwords, stemming_type="porter")',
            )
        )

    @weight(2.5)
    @visibility("after_due_date")
    @number("1.6.2")
    def test_162_statistics_tokenization_fancy_yesStopping_and_stemming_porter(self):
        self.zipf_tester(
            prerequisite=(
                "tokenization_fancy_yesStopping_and_stemming_porter_{store_type}",
                'tokenization(tokenize_type="fancy", stopwords=stopwords, stemming_type="porter")',
            )
        )
