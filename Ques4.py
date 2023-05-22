import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the sentences into words
    word_tokens = [word_tokenize(sentence) for sentence in sentences]

    # Convert words to lowercase
    word_tokens = [[word.lower() for word in words] for words in word_tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = [[word for word in words if word not in stop_words] for words in word_tokens]

    # Stem the words
    stemmer = PorterStemmer()
    word_tokens = [[stemmer.stem(word) for word in words] for words in word_tokens]

    return word_tokens

def calculate_similarity_matrix(word_tokens):
    # Flatten the word tokens
    all_words = [word for words in word_tokens for word in words]

    # Create a frequency distribution of the words
    word_freq = nltk.FreqDist(all_words)

    # Build the similarity matrix
    similarity_matrix = np.zeros((len(word_tokens), len(word_tokens)))
    for i in range(len(word_tokens)):
        for j in range(len(word_tokens)):
            if i != j:
                similarity_matrix[i][j] = calculate_sentence_similarity(word_tokens[i], word_tokens[j], word_freq)

    return similarity_matrix

def calculate_sentence_similarity(sentence1, sentence2, word_freq):
    # Calculate term frequency for the words in the sentences
    tf_sentence1 = [sentence1.count(word) / len(sentence1) for word in sentence1]
    tf_sentence2 = [sentence2.count(word) / len(sentence2) for word in sentence2]

    # Calculate inverse document frequency for the words in the sentences
    idf_sentence1 = [np.log(len(sentence1) / (word_freq[word] + 1)) for word in sentence1]
    idf_sentence2 = [np.log(len(sentence2) / (word_freq[word] + 1)) for word in sentence2]

    # Create sentence vectors based on TF-IDF
    vector1 = np.array(tf_sentence1)[:, np.newaxis] * np.array(idf_sentence1)[:, np.newaxis]
    vector2 = np.array(tf_sentence2)[:, np.newaxis] * np.array(idf_sentence2)[:, np.newaxis]

    # Calculate cosine similarity between the sentence vectors
    similarity = cosine_similarity(vector1, vector2)[0][0]

    return similarity

def generate_summary(text, num_sentences):
    # Preprocess the text
    word_tokens = preprocess_text(text)

    # Calculate the similarity matrix
    similarity_matrix = calculate_similarity_matrix(word_tokens)

    # Convert the similarity matrix to a network graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Calculate the sentence scores using PageRank algorithm (TextRank)
    scores = nx.pagerank(graph)

    # Sort the sentences by score in descending order
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sent_tokenize(text))), reverse=True)

    # Select the top N sentences as the summary
    summary = ' '.join([sentence for _, sentence in ranked_sentences[:num_sentences]])

    return summary

# Read the text file
with open('intro_video.txt', 'r') as file:
    text = file.read()

# Generate the summary with 3 sentences
summary = generate_summary(text, num_sentences=3)

# Print the summary
print(summary)

