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
    # And return the list of words
    return [word.lower() for word in document.split() if word.lower() not in nltk.corpus.stopwords.words("english")]
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
    file_idfs = dict()

    # For each file, compute the sum of tf-idfs for each word in the query
    for file in files.keys():
        # For each word in the query, determine the number of occurrences in the document ("files[file]")
        # And multiply by the idf value ("idfs[word]"), this is the word's tf-idf score
        # Sum all these values for the file
        tf_idf_sum = sum([files[file].count(word) * idfs[word] for word in query])
        # Map the tf idf sum to the filename (so we can sort them later)
        file_idfs[file] = tf_idf_sum

    # Sort the files by tf_idf_sum and get the top n files
    # The lambda function used here can be read as -> given a variable "item", return "item[1]"
    # Note that the .items() function gives us a list of tuples representing each k-v pair
    # So passing each item into the lambda function gives us the tf_idf_sum (second item in the tuple),
    # i.e. we're asking the sorted function to sort by the tf_idf_sum of each tuple
    # Sorted returns the sorted list of .items() tuples
    top_files = sorted(file_idfs.items(), key=lambda item: item[1], reverse=True)
    
    # Return the filenames for the top n files
    return [item[0] for item in top_files[:n]]
    # raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_idfs = dict()

    # For each sentence, compute the sum of idfs for each word in the query and the query term density
    for sentence in sentences.keys():
        # Sum the IDF values of all words in the query that appears in the sentence ("if word in sentences[sentence]")
        idf_sum = sum([idfs[word] for word in query if word in sentences[sentence]])
        # Calculate the "query term density" for the sentence
        # This is the proportion of words in the sentence that are also words in the query
        query_term_density = len([word for word in sentences[sentence] if word in query]) / len(sentences[sentence])
        # Map the idf sum and query_term_density to the filename (so we can sort them later)
        sentence_idfs[sentence] = (idf_sum, query_term_density)

    # Sort the sentences by idf_sum and query_term_density and get the top n sentences
    # Here, our sorting key is a tuple, so the sorted function sorts by the first value (in this case, idf_sum) then by the second value
    top_sentences = sorted(sentence_idfs.items(), key=lambda item: item[1], reverse=True)
    
    # Return the top n sentences
    return [item[0] for item in top_sentences[:n]]
    # raise NotImplementedError


if __name__ == "__main__":
    main()
