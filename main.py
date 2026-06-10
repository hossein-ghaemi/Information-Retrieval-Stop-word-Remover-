import re
import os

from test_wrapper import (
    load_documents_from_url,
    linear_boolean_search,
)
from my_module import remove_stop_words
def load_stop_words(filename=os.path.join("public_tests", "englishST.txt")):
    with open(filename, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def print_menu():
    print("\n" + "=" * 50)
    print("Information Retrieval System")
    print("=" * 50)
    print("1. Download Collection from URL")
    print("2. Search Documents")
    print("3. Apply stopword removal")
    print("4. Show documents")
    print("5. Export filtered documents to file")
    print("0. Exit")
    print("=" * 50)


def load_collection_ui():
    print("\nLOAD COLLECTION")

    url = input("URL: ")
    author = input("Author: ")
    origin = input("Origin: ")

    start_line = int(input("start line: "))
    end_line = int(input("end line: "))

    print("Regex must contain:")
    print("Group 1 = title")
    print("Group 2 = document text")

    search_pattern = input("Regex pattern: ")

    try:
        pattern = re.compile(search_pattern, re.DOTALL)

        collection = load_documents_from_url(
            url=url,
            author=author,
            origin=origin,
            start_line=start_line,
            end_line=end_line,
            search_pattern=pattern
        )

        print(f"\nLoaded {len(collection)} documents.")
        return collection

    except Exception as e:
        print("\nError while loading collection:", e)
        return []


def search_ui(collection):
    if not collection:
        print("No collection loaded.")
        return

    term = input("Search term: ")

    use_filtered = input("Search filtered terms? (y/n): ").lower() == "y"

    results = linear_boolean_search(term, collection, stopword_filtered=use_filtered)

    matches = 0

    for score, doc in results:
        if score == 1:
            matches += 1
            print("\n----------------------------")
            print("ID:", doc.document_id)
            print("Title:", doc.title)
            print("Author:", doc.author)
            print("Preview:", doc.raw_text[:100])
            print("----------------------------")

    print(f"\nMatching documents: {matches}")


def stopword_removal_ui(collection):
    if not collection:
        print("No collection loaded")
        return

    try:
        stopwords = load_stop_words()
        for doc in collection:
            # remove_stop_words takes a list, returns a list
            doc._filtered_terms = remove_stop_words(doc.terms, stopwords)
        print(f"Stopword removal completed using {len(stopwords)} stopwords.")
    except Exception as e:
        print("Error:", e)

def show_documents_ui(collection):
    if not collection:
        print("No collection loaded")
        return

    print("\nDOCUMENTS\n")

    for doc in collection:
        print(doc)


def export_filtered_documents(collection):
    if not collection:
        print("No collection loaded")
        return

    if not any(doc._filtered_terms for doc in collection):
        print("No filtered terms found. Please run stopword removal first")
        return

    filename = input("ENTER OUTPUT FILENAME (E.G. FILTERED_DOCUMENTS.TXT ")
    if not filename:
        filename = " filtered_documents.txt"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            for doc in collection:
                f.write("=" * 60 + "\n")
                f.write(f"ID:   D{str(doc.document_id).zfill(3)}\n")
                f.write(f"Title:  {doc.title}\n")
                f.write(f"Author: {doc.author}\n")
                f.write(f"Origin: {doc.origin}\n")
                f.write("-" * 60 + "\n")
                f.write("Original terms:\n")
                f.write(" ".join(doc.terms) + "\n")
                f.write("-" * 60 + "\n")
                f.write("Filtered terms (stopwords removed):\n")
                f.write(" ".join(doc._filtered_terms) + "\n")
                f.write("\n")

            print(f"Exported {len(collection)} documents to '{filename}'.")
    except Exception as e:
        print("Error while exporting:",e)

def main():
    collection = []

    while True:
        print_menu()
        choice = input("Choice: ")

        if choice == "1":
            collection = load_collection_ui()

        elif choice == "2":
            search_ui(collection)

        elif choice == "3":
            stopword_removal_ui(collection)

        elif choice == "4":
            show_documents_ui(collection)

        elif choice == "5":
            export_filtered_documents(collection)

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()