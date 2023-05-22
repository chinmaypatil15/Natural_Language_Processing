import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load the word counts from the CSV file
df = pd.read_csv('word_counts.csv')

# Combine all words into a single string
all_words = ' '.join(df['Word'])

# Tokenize the text
tokens = word_tokenize(all_words)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token.lower() not in stop_words]

# Create a dictionary from the tokens
dictionary = corpora.Dictionary([tokens])

# Convert the tokens to a bag-of-words representation
corpus = [dictionary.doc2bow(tokens)]

# Ensure the length of corpus matches the length of the DataFrame's index
corpus = corpus * len(df)

# Perform topic modeling using Latent Dirichlet Allocation (LDA)
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=5)

# Print the most probable words for each topic
for topic_num, topic_words in lda_model.print_topics(num_words=5):
    print(f"Topic {topic_num}: {topic_words}")

# Get the dominant topic for each word
dominant_topics = [lda_model.get_document_topics(doc) for doc in corpus]
dominant_topic_nums = [max(topics, key=lambda x: x[1])[0] for topics in dominant_topics]

# Add dominant topic information to the DataFrame
df['Dominant Topic'] = dominant_topic_nums

# Save the updated DataFrame to a new CSV file
df.to_csv('word_topics.csv', index=False)
