import pandas as pd
from googleapiclient.discovery import build

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


# Set up the YouTube Data API
api_key = 'AIzaSyBGLGRo5QIZtOGB2zUA18zndSQHVX9uLmc'
youtube = build('youtube', 'v3', developerKey=api_key)

# Fetch the comments
video_id = 'bPrmA1SEN2k'
comments = []

next_page_token = None
while True:
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100,
        pageToken=next_page_token
    ).execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    next_page_token = response.get('nextPageToken')

    if not next_page_token:
        break

# Store comments in a CSV file
df = pd.DataFrame({'comments': comments})
df.to_csv('comments.csv', index=False)


# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load the comments from the CSV file
df = pd.read_csv('comments.csv')

# Combine all comments into a single string
all_comments = ' '.join(df['comments'])

# Tokenize the text
tokens = word_tokenize(all_comments)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token.lower() not in stop_words]

# Count the frequency of each keyword
keyword_counts = Counter(tokens)

# Print the top 10 most common keywords
print(keyword_counts.most_common(10))