import nltk
import sys
import string
import math
import os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    # Create a dictionary
    files = dict()

    # Access the directory and, for each file, update them to the dictionary as {filename: content} key-value pair
    # List of files
    file_list = os.listdir(directory)
    # For each file, read contents into a string and add to the dictionary
    for file_name in file_list:
        with open(os.path.join(directory, file_name)) as file:
            files[file_name] = file.read()

    # return the dictionary
    return files
    # raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Download the nltk stopwords resource
    # nltk.download("stopwords")

    # Remove all punctuation marks from the text (from string.punctuation)
    # ord() returns the integer value of a unicode character, which we use to search the input string
    document = document.translate({ord(punct): None for punct in string.punctuation})

    # Splice the string into a list of lower case words (delimited by spaces)
    words = [word.lower() for word in document.split() if word.lower() not in nltk.corpus.stopwords.words("english")]
    
    # Return the list of words
    return words
    # raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_values = dict()

    # Count the number of documents (for use in the idf computation)
    total_documents = len(documents)

    # Create a list of all unique words from the documents by creating a set (which removes duplicate values) from each document's word list
    word_list = set()
    for document in documents:
        word_list.update(documents[document])

    # Compute idf for each word; idf is computed as the natural logarithm of the total number of documents / number of documents containing the word
    for word in word_list:
        # Count the number of documents that contain the word
        docs_with_word = len([document for document in documents if word in documents[document]])
        # Compute the idf and add to the output dictionary
        idf = math.log(total_documents / docs_with_word)
        idf_values[word] = idf

    return idf_values
    # raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
