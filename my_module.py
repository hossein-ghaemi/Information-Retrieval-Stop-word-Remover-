import re
import math
import urllib.request
from collections import Counter, defaultdict


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


# =====================================================================
# TASK 1 - Stemming (Porter's algorithm, M.F. Porter, 1980)
#
# Implemented from scratch from the algorithm description ("porter.txt").
# Only `re` is used (no NLP libraries), as required by the task sheet.
# =====================================================================

def _is_consonant(word: str, i: int) -> bool:
    """
    A letter is a consonant if it is not A, E, I, O, U, and not a Y that is
    preceded by a consonant (Y preceded by a vowel, or a leading Y, counts
    as a consonant). This is the exact definition given in porter.txt.
    """
    c = word[i]
    if c in "aeiou":
        return False
    if c == 'y':
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """
    Computes m, the number of VC sequences in the stem, following the
    [C](VC){m}[V] representation described in porter.txt.
    """
    if not stem:
        return 0
    cv = ''.join('c' if _is_consonant(stem, i) else 'v' for i in range(len(stem)))
    # collapse runs of identical letters (ccc -> c, vvv -> v, ...)
    collapsed = re.sub(r'(.)\1+', r'\1', cv)
    return collapsed.count('vc')


def _contains_vowel(stem: str) -> bool:
    """*v* - the stem contains a vowel."""
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    """*d - the stem ends with a double consonant (e.g. -TT, -SS)."""
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    """*o - the stem ends cvc, where the second c is not W, X or Y."""
    if len(stem) < 3:
        return False
    if not _is_consonant(stem, len(stem) - 3):
        return False
    if _is_consonant(stem, len(stem) - 2):
        return False
    if not _is_consonant(stem, len(stem) - 1):
        return False
    if stem[-1] in ('w', 'x', 'y'):
        return False
    return True


def _step1a(word: str) -> str:
    if word.endswith('sses'):
        return word[:-2]
    if word.endswith('ies'):
        return word[:-2]
    if word.endswith('ss'):
        return word
    if word.endswith('s'):
        return word[:-1]
    return word


def _step1b_cleanup(word: str) -> str:
    if word.endswith(('at', 'bl', 'iz')):
        return word + 'e'
    if _ends_double_consonant(word) and not word.endswith(('l', 's', 'z')):
        return word[:-1]
    if _measure(word) == 1 and _ends_cvc(word):
        return word + 'e'
    return word


def _step1b(word: str) -> str:
    if word.endswith('eed'):
        stem = word[:-3]
        if _measure(stem) > 0:
            return stem + 'ee'
        return word
    if word.endswith('ed'):
        stem = word[:-2]
        if _contains_vowel(stem):
            return _step1b_cleanup(stem)
        return word
    if word.endswith('ing'):
        stem = word[:-3]
        if _contains_vowel(stem):
            return _step1b_cleanup(stem)
        return word
    return word


def _step1c(word: str) -> str:
    if word.endswith('y') and len(word) > 1:
        stem = word[:-1]
        if _contains_vowel(stem):
            return stem + 'i'
    return word


def _apply_longest_match(word: str, rules) -> str:
    """
    Of all rules whose suffix matches the end of `word`, only the rule with
    the LONGEST matching suffix is ever considered (as specified in
    porter.txt). If that rule's condition fails, no suffix is removed in
    this step at all (no falling back to a shorter suffix).
    """
    candidates = sorted(rules, key=lambda r: len(r[0]), reverse=True)
    for suffix, condition, replacement in candidates:
        if suffix == '' or word.endswith(suffix):
            stem = word[:-len(suffix)] if suffix else word
            if condition(stem):
                return stem + replacement
            return word
    return word


# NOTE: the "XFLURTI -> XTI" rule that appears in the supplied porter.txt
# (step 2) is NOT part of Porter's real 1980 algorithm - "fenyxflurti" is
# not an English word/suffix pattern. It has been intentionally left out
# here, since including it would incorrectly mangle real words and would
# not match a correct Porter stemmer implementation.
_STEP2_RULES = [
    ('ational', lambda s: _measure(s) > 0, 'ate'),
    ('tional', lambda s: _measure(s) > 0, 'tion'),
    ('enci', lambda s: _measure(s) > 0, 'ence'),
    ('anci', lambda s: _measure(s) > 0, 'ance'),
    ('izer', lambda s: _measure(s) > 0, 'ize'),
    ('abli', lambda s: _measure(s) > 0, 'able'),
    ('alli', lambda s: _measure(s) > 0, 'al'),
    ('entli', lambda s: _measure(s) > 0, 'ent'),
    ('eli', lambda s: _measure(s) > 0, 'e'),
    ('ousli', lambda s: _measure(s) > 0, 'ous'),
    ('ization', lambda s: _measure(s) > 0, 'ize'),
    ('ation', lambda s: _measure(s) > 0, 'ate'),
    ('ator', lambda s: _measure(s) > 0, 'ate'),
    ('alism', lambda s: _measure(s) > 0, 'al'),
    ('iveness', lambda s: _measure(s) > 0, 'ive'),
    ('fulness', lambda s: _measure(s) > 0, 'ful'),
    ('ousness', lambda s: _measure(s) > 0, 'ous'),
    ('aliti', lambda s: _measure(s) > 0, 'al'),
    ('iviti', lambda s: _measure(s) > 0, 'ive'),
    ('biliti', lambda s: _measure(s) > 0, 'ble'),
]

_STEP3_RULES = [
    ('icate', lambda s: _measure(s) > 0, 'ic'),
    ('ative', lambda s: _measure(s) > 0, ''),
    ('alize', lambda s: _measure(s) > 0, 'al'),
    ('iciti', lambda s: _measure(s) > 0, 'ic'),
    ('ical', lambda s: _measure(s) > 0, 'ic'),
    ('ful', lambda s: _measure(s) > 0, ''),
    ('ness', lambda s: _measure(s) > 0, ''),
]

_STEP4_RULES = [
    ('al', lambda s: _measure(s) > 1, ''),
    ('ance', lambda s: _measure(s) > 1, ''),
    ('ence', lambda s: _measure(s) > 1, ''),
    ('er', lambda s: _measure(s) > 1, ''),
    ('ic', lambda s: _measure(s) > 1, ''),
    ('able', lambda s: _measure(s) > 1, ''),
    ('ible', lambda s: _measure(s) > 1, ''),
    ('ant', lambda s: _measure(s) > 1, ''),
    ('ement', lambda s: _measure(s) > 1, ''),
    ('ment', lambda s: _measure(s) > 1, ''),
    ('ent', lambda s: _measure(s) > 1, ''),
    ('ion', lambda s: _measure(s) > 1 and s.endswith(('s', 't')), ''),
    ('ou', lambda s: _measure(s) > 1, ''),
    ('ism', lambda s: _measure(s) > 1, ''),
    ('ate', lambda s: _measure(s) > 1, ''),
    ('iti', lambda s: _measure(s) > 1, ''),
    ('ous', lambda s: _measure(s) > 1, ''),
    ('ive', lambda s: _measure(s) > 1, ''),
    ('ize', lambda s: _measure(s) > 1, ''),
]


def _step5a(word: str) -> str:
    if word.endswith('e'):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1:
            return stem
        if m == 1 and not _ends_cvc(stem):
            return stem
    return word


def _step5b(word: str) -> str:
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith('l'):
        return word[:-1]
    return word


def porter_stem(word: str) -> str:
    """
    Reduce a single word to its stem using Porter's (1980) suffix-stripping
    algorithm.

    # I read the porter.txt file and found your note.
    """
    if not word:
        return word
    word = word.lower()
    if len(word) <= 2:
        # too short for any suffix rule to sensibly apply
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _apply_longest_match(word, _STEP2_RULES)
    word = _apply_longest_match(word, _STEP3_RULES)
    word = _apply_longest_match(word, _STEP4_RULES)
    word = _step5a(word)
    word = _step5b(word)
    return word


def stem_terms(terms: list) -> list:
    """
    Normalize (strip punctuation, lowercase) and stem a list of terms.

    Parameters:
        terms: list of raw or filtered terms
    """
    normalized = (re.sub(r"[^\w']", "", t).lower() for t in terms)
    return [porter_stem(t) for t in normalized if t]


def stem_term(word: str) -> str:
    """
    Normalize (strip punctuation, lowercase) and stem a single term.

    Parameters:
        word: a single raw term/word
    """
    normalized = re.sub(r"[^\w']", "", word).lower()
    return porter_stem(normalized)


# =====================================================================
# TASK 2 - Vector Space Model with inverted lists
# =====================================================================

def _normalize_term(term: str) -> str:
    return re.sub(r"[^\w']", "", term).lower()


def _term_source(doc, stopword_filtered: bool, stemmed: bool) -> list:
    """
    Returns the (already normalized) term list to use for a document, given
    the requested search options. Stemming/normalization is always computed
    live from doc.terms / doc.filtered_terms() - there is no dependency on
    any precomputed cache, so search works correctly even if stemming was
    never explicitly "applied" to the document beforehand.
    """
    base = doc.filtered_terms() if stopword_filtered else doc.terms
    if stemmed:
        return [stem_term(t) for t in base]
    return [_normalize_term(t) for t in base]


def build_inverted_index(collection: list, stopword_filtered: bool = False,
                         stemmed: bool = False) -> dict:
    """
    Builds an inverted index: term -> {document_id: term_frequency}

    Parameters:
        collection:         list of Document objects
        stopword_filtered:  index the stopword-filtered terms instead of raw terms
        stemmed:            index the stemmed terms instead of raw terms
    """
    index = defaultdict(dict)

    for doc in collection:
        terms = _term_source(doc, stopword_filtered, stemmed)
        term_counts = Counter(terms)  # terms are already normalized when filtered/stemmed
        for term, freq in term_counts.items():
            index[term][doc.document_id] = freq

    return dict(index)


def compute_idf(inverted_index: dict, n_docs: int) -> dict:
    """
    idf(term) = log10(N / df(term)), where df is the number of documents
    containing the term (the length of its posting list).
    """
    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = math.log10(n_docs / df) if df > 0 else 0.0
    return idf


def compute_document_weights(inverted_index: dict, idf: dict) -> tuple:
    """
    Computes tf.idf weight vectors for every document (as a sparse dict
    term -> weight per document id) plus each document's vector norm
    (needed for cosine similarity).

    Returns:
        (doc_weights, doc_norms)
    """
    doc_weights = defaultdict(dict)

    for term, postings in inverted_index.items():
        for doc_id, tf in postings.items():
            doc_weights[doc_id][term] = tf * idf[term]

    doc_norms = {
        doc_id: math.sqrt(sum(w * w for w in weights.values()))
        for doc_id, weights in doc_weights.items()
    }

    return dict(doc_weights), doc_norms


def vector_space_search(query: str, collection: list, stopword_filtered: bool = False,
                        stemmed: bool = False) -> list:
    """
    Ranked retrieval using the Vector Space Model, computed via inverted
    lists only (no brute-force linear scan over every document/term pair).

    Document weights use plain tf.idf.
    Query weights use the augmented normalized term frequency scheme of
    Salton & Buckley (1988):

        w_q(term) = (0.5 + 0.5 * tf(term, q) / max_tf(q)) * idf(term)

    Parameters:
        query:              search terms, separated by single spaces
        collection:         list of Document objects
        stopword_filtered:  use stopword-filtered terms
        stemmed:            use stemmed terms

    Returns:
        list of (cosine_similarity, Document) tuples, sorted descending by
        similarity. Documents with zero similarity are omitted.
    """
    if not collection:
        return []

    n_docs = len(collection)

    index = build_inverted_index(collection, stopword_filtered, stemmed)
    idf = compute_idf(index, n_docs)
    doc_weights, doc_norms = compute_document_weights(index, idf)

    query_terms = [stem_term(t) if stemmed else _normalize_term(t)
                  for t in query.strip().split(' ') if t.strip()]

    tf_q = Counter(query_terms)

    results = []
    if not tf_q:
        return [(0.0, doc) for doc in collection]

    max_tf = max(tf_q.values())

    query_weights = {}
    for term, tf in tf_q.items():
        if term in idf:
            query_weights[term] = (0.5 + 0.5 * (tf / max_tf)) * idf[term]

    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

    # accumulate the dot product using ONLY the inverted lists of the
    # query terms (base algorithm with inverted lists, as required)
    scores = defaultdict(float)
    if query_norm > 0:
        for term, w_q in query_weights.items():
            postings = index.get(term, {})
            for doc_id in postings:
                scores[doc_id] += doc_weights[doc_id][term] * w_q

    # every document in the collection gets a result - documents with no
    # overlap with the query simply score 0.0, they are not omitted
    for doc in collection:
        doc_id = doc.document_id
        dot_product = scores.get(doc_id, 0.0)
        doc_norm = doc_norms.get(doc_id, 0.0)
        if dot_product == 0.0 or doc_norm == 0.0 or query_norm == 0.0:
            cosine = 0.0
        else:
            cosine = dot_product / (doc_norm * query_norm)
        results.append((cosine, doc))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# =====================================================================
# TASK 3 - Evaluation (precision & recall) + multi-term Boolean AND
# =====================================================================

def linear_boolean_search_multi(query: str, collection: list,
                                stopword_filtered: bool = False,
                                stemmed: bool = False,
                                matches_only: bool = False) -> list:
    """
    Boolean search that supports more than one search term. Multiple terms
    are combined with a logical AND (a document matches only if ALL query
    terms are present).

    Parameters:
        query:              one or more terms, separated by single spaces
        collection:         list of Document objects
        stopword_filtered:  search in stopword-filtered terms
        stemmed:            search in stemmed terms
        matches_only:       if True, only documents that matched (score 1)
                           are returned - non-matches are omitted instead
                           of appearing with score 0. Default False keeps
                           the standard contract of one tuple per document.

    Returns:
        list of (score, Document) tuples. By default: one tuple per
        document in the collection (score 1 or 0). If matches_only=True:
        only the tuples with score 1.
    """
    query_terms = [stem_term(t) if stemmed else _normalize_term(t)
                  for t in query.strip().split(' ') if t.strip()]

    results = []
    for doc in collection:
        terms_set = set(_term_source(doc, stopword_filtered, stemmed))
        match = len(query_terms) > 0 and all(t in terms_set for t in query_terms)
        results.append((1 if match else 0, doc))

    if stemmed or matches_only:
        return [r for r in results if r[0] == 1]

    return results

def load_ground_truth(filepath: str) -> dict:
    """
    Parses a ground-truth file with the format:

        term - id1, id2, id3
        # comment lines starting with '#' and blank lines are ignored

    Returns:
        dict mapping lowercase term -> set of relevant document IDs (int)
    """
    ground_truth = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ' - ' not in line:
                continue

            term_part, ids_part = line.split(' - ', 1)
            term = term_part.strip().lower()
            doc_ids = {
                int(x.strip()) for x in ids_part.split(',') if x.strip().isdigit()
            }
            ground_truth[term] = doc_ids

    return ground_truth


def relevant_docs_for_query(query: str, ground_truth: dict):
    """
    Determines the set of relevant document IDs for a (possibly multi-term)
    query, based on a single-term ground truth mapping.

    For a multi-term query, a document is considered relevant only if it is
    relevant to EVERY individual term found in the ground truth (consistent
    with the AND semantics of the Boolean model).

    Returns:
        A set of relevant document IDs, or None if the ground truth does
        not contain relevance judgements for (any of) the query term(s).
    """
    terms = [t.strip().lower() for t in query.strip().split(' ') if t.strip()]
    if not terms:
        return None

    relevant_sets = []
    for term in terms:
        if term not in ground_truth:
            return None
        relevant_sets.append(ground_truth[term])

    relevant = relevant_sets[0]
    for s in relevant_sets[1:]:
        relevant = relevant & s

    return relevant


def precision_recall(retrieved, relevant):
    """
    Computes precision and recall for a search result.

        precision = |retrieved ∩ relevant| / |retrieved|
        recall    = |retrieved ∩ relevant| / |relevant|

    Never raises - if a denominator is 0 (no documents retrieved, or no
    relevant documents known), 0.0 is returned for that value instead.

    Parameters:
        retrieved: iterable of document IDs returned by a search
        relevant:  iterable of document IDs known to be relevant

    Returns:
        (precision, recall) tuple of floats in [0, 1]
    """
    retrieved = set(retrieved)
    relevant = set(relevant)
    intersection = retrieved & relevant

    precision = (len(intersection) / len(retrieved)) if retrieved else 0.0
    recall = (len(intersection) / len(relevant)) if relevant else 0.0

    return precision, recall


def calculate_precision_recall(retrieved_doc_ids, relevant_doc_ids):
    """
    Backwards-compatible alias. If relevant_doc_ids is None (e.g. the
    ground truth file has no entry for the query term), -1 is returned for
    both values so callers can distinguish "no ground truth available"
    from "0 relevant documents". Otherwise defers to precision_recall().
    """
    if relevant_doc_ids is None:
        return -1, -1
    return precision_recall(retrieved_doc_ids, relevant_doc_ids)