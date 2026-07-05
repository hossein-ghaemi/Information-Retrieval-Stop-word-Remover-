# Information Retrieval - Practical Task 2 & 3
# Wrapper for Unit Tests
# Version 1.1 (2025-06-04)
#
# PR03 additions (stem_term, vector_space_search, load_ground_truth,
# precision_recall) match the function names/signatures confirmed by the
# official test_pr03_t1/t2/t3.py test files.

# You must implement this file so that the test suite can run your code.
# This file acts as a bridge between your individual implementation and the expected interface.

# You are free to organize your own code however you want - but make sure
# that the following three functions are importable and behave as specified below.

from document import Document
from re import Pattern


def remove_stopwords_by_list(doc: Document, stopwords: set[str]):
    """
    Remove stopwords from the given document and store the result in doc.filtered_terms.

    Your implementation must:
    - Take a Document object and a set of stop words
    - Filter out the stop words from doc.terms
    - Store the cleaned list in doc.filtered_terms
    - Leave doc.terms and doc.raw_text unchanged

    Parameters:
        doc: The document to clean
        stopwords: The stop words to remove
    """

    # The following code is an example. You may replace it how you see fit:
    from my_module import remove_stop_words
    doc._filtered_terms = remove_stop_words(doc.terms, stopwords)


def remove_stopwords_by_frequency(doc, collection: list[Document], common_frequency: float, rare_frequency: float):
    """
    Remove stopwords from the given document and store the result in doc.filtered_terms.

    Your implementation must:
    - Take a Document object and a set of stop words
    - Filter out the stop words from doc.terms
    - Store the cleaned list in doc.filtered_terms
    - Leave doc.terms and doc.raw_text unchanged

    Parameters:
        doc: The document to clean
        collection: A collection of documents to use as a reference
        common_frequency: The frequency at which a term is "too common" to hold meaningful semantics.
        rare_frequency: The frequency at which a term is "too rare" to help finding a document.
    """

    # The following code is an example. You may replace it how you see fit:
    # from my_module import remove_stopwords
    # remove_stopwords_by_frequency(doc, collection, common_frequency, rare_frequency)

    from my_module import remove_stop_words_by_frequency
    doc._filtered_terms = remove_stop_words_by_frequency(doc.terms, collection, low_freq=rare_frequency,
                                                         high_freq=common_frequency)


def load_documents_from_url(url: str, author: str, origin: str, start_line: int, end_line: int,
                            search_pattern: Pattern[str] = r'([^\n]+)\n\n(.*?)(?=\n{5}(?=[^\n]+\n\n)|$)') -> list[
    Document]:
    """
    Download a text from the given URL, extract stories/chapters and return them as Document objects.

    Your implementation must:
    - Download the text file at the given URL
    - Split it into individual stories (each a Document)
    - Fill in all Document fields: title, raw_text, terms, author, origin (URL)
    - Return a list of Document instances

    Parameters:
        url (str): The URL to the Project Gutenberg text file
        author (str): The author name to assign to each document
        origin: The title of the containing collection, to assign to each document
        start_line: Line number from where to start searching
        end_line: Line number until which to search
        search_pattern: RE pattern where the 1st capture group contains the title and the 2nd the text of the document


    Returns:
        list[Document]: List of parsed documents
    """

    # The following code is an example. You may replace it how you see fit:
    from my_module import load_collection_from_url
    return load_collection_from_url(url, search_pattern, start_line, end_line, author, origin)


def linear_boolean_search(term, collection, stopword_filtered=False, stemmed=False, matches_only=False):
    """
    Search a given collection of documents for all documents that contain a given term, using a simple Boolean model.

    Your implementation must:
    - Take a term (or several terms separated by single spaces) and a list of Document objects
    - Combine multiple terms with a logical AND
    - Return a list of relevant documents, more specifically a list of tuples (relevance score, document)

    Parameters:
        term: The search term(s), space-separated
        collection: A collection of documents to search in
        stopword_filtered: If true, stopwords are not considered in the search
        stemmed: If true, search is performed against stemmed terms (Task 1)
        matches_only: If true, only score-1 documents are returned (non-matches
                     omitted). Default False returns one tuple per document.
    """

    # The following code is an example. You may replace it how you see fit:
    from my_module import linear_boolean_search_multi
    return linear_boolean_search_multi(term, collection, stopword_filtered, stemmed, matches_only)


# ---------------------------------------------------------------------
# TASK 1 - Stemming
# ---------------------------------------------------------------------

def stem_term(word):
    """
    Reduce a single term to its stem using Porter's (1980) algorithm.

    Your implementation must:
    - Take a single term (string)
    - Normalize it (case/punctuation) and return its stem as a string
    - Not depend on any other function having been called first - it must
      work standalone, e.g. for use inside search functions when stemmed=True

    Parameters:
        word: The term to stem
    """
    from my_module import stem_term as porter_stem_term
    return porter_stem_term(word)


# ---------------------------------------------------------------------
# TASK 2 - Vector Space Model
# ---------------------------------------------------------------------

def vector_space_search(query, collection, stopword_filtered=False, stemmed=False):
    """
    Search a given collection of documents using the Vector Space Model
    with tf.idf weights, computed via inverted lists.

    Your implementation must:
    - Take a query (terms separated by single spaces) and a list of Document objects
    - Build/use an inverted index rather than a linear scan
    - Weight documents with tf.idf
    - Weight the query according to Salton/Buckley (1988)
    - Return a list of (cosine similarity score, document) tuples for
      EVERY document in the collection (score 0.0 for non-matches), ranked
      descending by score

    Parameters:
        query: The search terms, space-separated
        collection: A collection of documents to search in
        stopword_filtered: If true, search is performed against stopword-filtered terms
        stemmed: If true, search is performed against stemmed terms
    """
    from my_module import vector_space_search as vsm_search
    return vsm_search(query, collection, stopword_filtered, stemmed)


# ---------------------------------------------------------------------
# TASK 3 - Evaluation
# ---------------------------------------------------------------------

def load_ground_truth(filepath):
    """
    Load a ground truth file mapping search terms to sets of relevant
    document IDs.

    Parameters:
        filepath: path to the ground truth .txt file
    """
    from my_module import load_ground_truth as load_gt
    return load_gt(filepath)


def precision_recall(retrieved, relevant):
    """
    Compute precision and recall for a set of retrieved document IDs
    against a set of known-relevant document IDs.

    Your implementation must:
    - Work for queries containing more than one search term (the caller
      is responsible for building the retrieved/relevant sets accordingly)
    - Never crash - return 0.0 for a value that cannot be computed
      (e.g. division by zero)

    Parameters:
        retrieved: set/iterable of document IDs returned by a search
        relevant:  set/iterable of document IDs known to be relevant

    Returns:
        (precision, recall) tuple of floats in [0, 1]
    """
    from my_module import precision_recall as precision_recall_impl
    return precision_recall_impl(retrieved, relevant)