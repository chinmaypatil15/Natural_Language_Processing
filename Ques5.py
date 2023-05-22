import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
pd.options.mode.chained_assignment = None
df = pd.read_csv(r'G:\project\Natural_Language_Processing\LanguageDetection.csv', index_col=None)
print(df)

#display the number of texts available  for every class (language)
plt.figure(figsize = (10, 8))
sns.countplot(x=df['Language'])
plt.show()
print(df.isnull().sum())

#!pip install  neattext
import neattext.functions as nfx
#text cleaning fct
def Clean_Text(data,column):
     #convert text to lower
    data[column]=data[column].str.lower()
    #replace \n and s with space
    data[column].replace(r'\s+|\\n', ' ',regex=True, inplace=True)
    #remove userhandles
    data[column]=data[column].apply(nfx.remove_userhandles)
    #remove urls
    data[column]=data[column].apply(nfx.remove_urls)
    #remove punctuations
    data[column]=data[column].apply(nfx.remove_punctuations)
    #remove special characters
    data[column]=data[column].apply(nfx.remove_special_characters)
    #remove emails
    data[column]=data[column].apply(nfx.remove_emails)
    #remove multiple space
    data[column]=data[column].apply(nfx.remove_multiple_spaces)
    #replace dates 1-2digits Mon 4digits
    data[column].replace(r'\d{1,2}\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|janv|juil|aot|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|January|February|March|April|May|June|July|August|September|October|November|December|avr|déc|févr|janv|juill|nov|oct|sept)\s\d{4}', ' ',regex=True, inplace=True)
    data[column].replace("(janv|\dh| h | \d |\d | \d|http|https|a35crasherait| d24d1minfriendly| \d+ \d+| \d+\d+)", "", regex=True, inplace=True)
    data[column].replace("  ", " ",regex=True, inplace=True)
    data[column].replace(r'(autres personnes|en rponse|rponse|en|[a-z][0-9][0-9][a-z]+|[0-9][0-9]+|[0,1,4,6,8]+|[0,1,4,6,8]+|[a-z][0,1,4,6,8])', ' ', regex=True, inplace=True)
    data[column].replace(r'avren|decn|fevren|janven|juilen|noven|octen|septen|avr|déc|févr|janv|juil|nov|oct|sept', ' ', regex=True, inplace=True)
    #replace /
    data[column].replace('\/', ' ',regex=True, inplace=True)
    #replace '
    data[column].replace('\'', ' ', regex=True, inplace=True)
    return data
dataset=Clean_Text(df,'Text')


import nltk
stopwords = set(nltk.corpus.stopwords.words('english')) | set(nltk.corpus.stopwords.words('french'))
dataset['Text'] = dataset['Text'].str.lower().apply(lambda x: ' '.join([word for word in str(x).split() if word not in stopwords]))
#delete empty rows
dataset = dataset[dataset['Text']!= '']
#reset data index
dataset=dataset.reset_index().drop('index',axis=1)

print(dataset)

