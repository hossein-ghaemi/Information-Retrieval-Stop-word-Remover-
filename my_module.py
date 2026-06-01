import re
import urllib.request
from collections import Counter


def load_collection_from_url(url: str, search_pattern, start_line: int,
                             end_line: int, author: str, origin: str) -> list:
    """
    Download a text file from a URL, extract stories using a regex pattern,
    and return them as a list of Document objects.

    Parameters:
        url:            URL pointing to a .txt file
        search_pattern: compiled regex — group(1) = title, group(2) = text
        start_line:     line number to start reading from (skip header)
        end_line:       line number to stop reading (skip footer), -1 = read all
        author:         author name to assign to each document
        origin:         collection title to assign to each document
    """
    from document import Document

    # Step 1: Download the file
    with urllib.request.urlopen(url) as response:
        raw_bytes = response.read()
        full_text = raw_bytes.decode("utf-8", errors="replace")

    # Step 2: Slice to start/end lines
    lines = full_text.splitlines()
    end = end_line if end_line > 0 else len(lines)
    sliced_lines = lines[start_line:end]

    # rejoin sliced content into one string for regex matching
    content = "\n".join(sliced_lines)

    # Step 3: Extract stories using the regex pattern
    matches = search_pattern.finditer(content)

    # Step 4: Build Document objects
    documents = []

    for doc_id, match in enumerate(matches):
        title = match.group(1).strip()
        text = match.group(2).strip()

        # remove line breaks from raw text
        raw_text = " ".join(text.splitlines())

        # tokenize — simple whitespace split, keep punctuation as-is
        terms = raw_text.split()

        doc = Document(
            document_id=doc_id,
            title=title,
            raw_text=raw_text,
            terms=terms,
            author=author,
            origin=origin
        )

        documents.append(doc)

    return documents


def linear_boolean_search(term: str, collection: list,
                          stopword_filtered: bool = False) -> list:
    """
    Linear scan through collection. Returns (score, doc) tuples for ALL documents.
    Score is 1 if term found, 0 otherwise.

    Parameters:
        term:               the search term
        collection:         list of Document objects
        stopword_filtered:  if True, search in filtered_terms instead of terms
    """
    term = term.lower().strip()
    results = []

    for doc in collection:
        # pick the right term list — no parentheses, filtered_terms is an attribute
        terms = doc.filtered_terms() if stopword_filtered else doc.terms

        # normalize each term for case-insensitive, punctuation-safe comparison
        normalized = [re.sub(r"[^\w']", "", t).lower() for t in terms]

        if term in normalized:
            results.append((1, doc))
        else:
            results.append((0, doc))

    return results


def remove_stop_words(terms: list, stopwords: set) -> list:
    """
    Filter stop words from a list of terms using a predefined stop word set.

    Parameters:
        terms:      list of terms to filter
        stopwords:  set of stop words (lowercase strings)
    """
    cleaned = []

    for term in terms:
        # normalize for comparison only — store original term if not a stopword
        # Regex removes all extra signs except the word. E.g: Hello, World! -> helloworld
        normalized = re.sub(r"[^\w']", "", term).lower()

        if normalized not in stopwords:
            cleaned.append(term)

    return cleaned


def remove_stop_words_by_frequency(terms: list, collection: list,
                                   low_freq: float, high_freq: float) -> list:
    """
    Filter stop words based on term frequency across the whole collection.
    Uses Crouch's frequency percentile method.

    Terms above high_freq percentile → too common  (e.g. "the", "and")
    Terms below low_freq percentile  → too rare    (not useful for retrieval)

    Parameters:
        terms:      terms of the document to filter
        collection: all documents — used to compute corpus frequencies
        low_freq:   lower percentile threshold (e.g. 0.01)
        high_freq:  upper percentile threshold (e.g. 0.99)
    """

    # Step 1: Count term frequencies across the whole corpus
    corpus_counter = Counter()

    for doc in collection:
        normalized_terms = [re.sub(r"[^\w']", "", t).lower() for t in doc.terms]
        corpus_counter.update(normalized_terms)

    # Step 2: Compute frequency thresholds from percentiles
    total_counts = sorted(corpus_counter.values())
    n = len(total_counts)

    low_threshold = total_counts[int(n * low_freq)]
    high_threshold = total_counts[int(min(n * high_freq, n - 1))]

    # Step 3: Build stop word set
    stopwords = set()

    for term, count in corpus_counter.items():
        if count <= low_threshold or count >= high_threshold:
            stopwords.add(term)

    # Step 4: Filter the document's terms
    cleaned = []

    for term in terms:
        normalized = re.sub(r"[^\w']", "", term).lower()
        if normalized not in stopwords:
            cleaned.append(term)

    return cleaned
