class BookwormError(Exception):
    code = "CORE_UNKNOWN1"
    message = "Unknown error"

    def __str__(self):
        return f"[{self.code}] {self.message}"


class BookNotFoundError(BookwormError):
    code = "CORE_BOOK1"
    message = "Book not found"


class InvalidBookIdError(BookwormError):
    code = "CORE_ID1"
    message = "Invalid book id"


class GutendexApiError(BookwormError):
    code = "CORE_GUTENDEX1"
    message = "Gutendex API error"


class DependencyError(BookwormError):
    code = "CORE_DEP1"
    message = "Missing dependency"


class EmptyTextError(BookwormError):
    code = "SUM_TEXT1"
    message = "Input text is empty"


class NoChapterFoundError(BookwormError):
    code = "SUM_CHAPTER1"
    message = "No chapter detected in document"


class EmptyChapterError(BookwormError):
    code = "SUM_CHAP_VALID1"
    message = "No valid chapter found"


class NoValidSentenceError(BookwormError):
    code = "SUM_SENTENCE1"
    message = "No valid sentence after filtering"


class EmptyVocabularyError(BookwormError):
    code = "SUM_VOCAB1"
    message = "TF-IDF generated an empty vocabulary"


class InvalidCosineMethodError(BookwormError):
    code = "SUM_COSINE1"
    message = "Unknown cosine similarity method"


class PageRankError(BookwormError):
    code = "SUM_PAGERANK1"
    message = "PageRank computation failed"


class TFEmptyDocumentError(BookwormError):
    code = "TFIDF_DOC1"
    message = "No chapters or documents found"


class TFVectorizerError(BookwormError):
    code = "TFIDF_VECT1"
    message = "TF-IDF computation failed"


class TFEmptyVocabularyError(BookwormError):
    code = "TFIDF_VOCAB1"
    message = "Empty vocabulary in TF-IDF"


class TFInvalidMethodError(BookwormError):
    code = "TFIDF_METHOD1"
    message = "Unknown TF-IDF method"


class BOWEmptyDocumentError(BookwormError):
    code = "BOW_DOC1"
    message = "No chapters or tokens found"


class BOWEmptyTokensError(BookwormError):
    code = "BOW_TOKEN1"
    message = "No valid tokens in chapter"


class BOWComputationError(BookwormError):
    code = "BOW_COMPUTE1"
    message = "Error during BOW computation"


class SimilarityDependencyError(BookwormError):
    code = "SIM_DEP1"
    message = "Missing dependency"


class SimilarityInvalidBookIdError(BookwormError):
    code = "SIM_ID1"
    message = "Invalid book id"


class EmptyVectorError(BookwormError):
    code = "SIM_VECTOR1"
    message = "Empty vector"


class EmbeddingError(BookwormError):
    code = "SBERT_EMBED1"
    message = "Embedding failed"


class SimilarityMatrixError(BookwormError):
    code = "SIM_MATRIX1"
    message = "Similarity matrix error"


class SimilarityComputationError(BookwormError):
    code = "SIM_COMPUTE1"
    message = "Similarity computation failed"


class InvalidTopicMethodError(BookwormError):
    code = "TOPIC_METHOD1"
    message = "Unknown topic method"


class TopicComputationError(BookwormError):
    code = "TOPIC_COMPUTE1"
    message = "Topic computation failed"


class TokenizationError(BookwormError):
    code = "PRE_TOKEN1"
    message = "Tokenization failed"


class NLTKResourceError(BookwormError):
    code = "PRE_NLTK1"
    message = "NLTK resources missing"


class StopwordsError(BookwormError):
    code = "PRE_STOP1"
    message = "Stopwords loading failed"


class PipelineError(BookwormError):
    code = "CARD_EXEC1"
    message = "Pipeline error"


class MetadataError(BookwormError):
    code = "CARD_META1"
    message = "Metadata error"


class EmptyPredictionError(BookwormError):
    code = "EVAL_PRED1"
    message = "No prediction results"


class NoTokensError(BookwormError):
    code = "LEX_TOKEN1"
    message = "No tokens found in text"


class DivisionByZeroError(BookwormError):
    code = "MATH_ZERO1"
    message = "Division by zero"


class ModelLoadingError(BookwormError):
    code = "MODEL_LOAD1"
    message = "Model loading failed"


class VectorizerError(BookwormError):
    code = "VEC_FAIL1"
    message = "Vectorizer error"


class BowComputationError(BookwormError):
    code = "BOW_COMPUTE2"
    message = "Error during BOW computation"


class EmptyDocumentError(BookwormError):
    code = "DOC_EMPTY1"
    message = "Document is empty"


class InvalidMethodError(BookwormError):
    code = "METHOD_INV1"
    message = "Invalid method specified"


class InvalidChapterError(BookwormError):
    code = "CHAP_INV1"
    message = "Invalid chapter structure"


class PreprocessingError(BookwormError):
    code = "PRE_FAIL1"
    message = "Preprocessing failed"
