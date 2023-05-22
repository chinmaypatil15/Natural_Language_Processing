import PyPDF2
import pandas as pd
from collections import Counter

# Open the PDF file
with open('research_paper.pdf', 'rb') as file:
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfFileReader(file)

    # Extract text from each page
    text = ''
    for page in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page).extractText()

# Remove unnecessary whitespace and newlines
text = text.replace('\n', ' ').strip()

# Split the text into words
words = text.split()

# Count the frequency of each word
word_counts = Counter(words)

# Store word counts in a DataFrame
df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])

# Sort the DataFrame by frequency in descending order
df = df.sort_values('Frequency', ascending=False)

# Save the word counts to a CSV file
df.to_csv('word_counts.csv', index=False)

# Print the most repeated word
most_common_word = df.iloc[0]['Word']
print(f"The most repeated word is: {most_common_word}")
