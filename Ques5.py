import matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

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
print(dataset)

import nltk
stopwords = set(nltk.corpus.stopwords.words('english')) | set(nltk.corpus.stopwords.words('french'))
dataset['Text'] = dataset['Text'].str.lower().apply(lambda x: ' '.join([word for word in str(x).split() if word not in stopwords]))
#delete empty rows
dataset = dataset[dataset['Text']!= '']
#reset data index
dataset=dataset.reset_index().drop('index',axis=1)

import nltk
dataset['tokenized_sents'] = dataset.apply(lambda row: nltk.word_tokenize(row['Text']), axis=1)
#remove words with less that 3 letters
def cleaner(dataset):
    for sentence in dataset.tokenized_sents:
        for token in sentence:
            if len(token) < 3  :
                sentence.remove(token)
    return dataset
dataset=cleaner(dataset)
print(dataset)

#After successfully removing noise from our tokenze we detokenize the sentences
from nltk.tokenize.treebank import TreebankWordDetokenizer
dataset['detokenized_sents'] = dataset.apply(lambda row: TreebankWordDetokenizer().detokenize(row['tokenized_sents']), axis=1)
dataset=dataset[dataset['detokenized_sents'].str.len()>=4]
print(dataset)

#Dominant voccabulary in Darija
#from wordcloud import WordCloud
from wordcloud import WordCloud
def plot_wordcloud(docx):
    mywordcloud=WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()

#liste of English keywords
language_list=dataset['Language'].unique().tolist()
English_list=dataset[dataset['Language']=='English']['detokenized_sents'].tolist()
English_docx=' '.join(English_list)
plot_wordcloud(English_docx)

#Dataset partition (trainning & testing data)
import numpy as np
from sklearn.model_selection import train_test_split

X=dataset['detokenized_sents']
y=dataset['Language']
#we used  80% for training data and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

from sklearn.feature_extraction.text import CountVectorizer
unigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
X_unigram_test_raw = unigramVectorizer.transform(X_test)

unigramFeatures = unigramVectorizer.get_feature_names()
print('Number of unigrams in training set:', len(unigramFeatures))

print(np.array(unigramFeatures))


# Distribution of uni-grams through the laguages
def train_lang_dict(X_raw_counts, y_train):
    lang_dict = {}
    for i in range(len(y_train)):
        lang = y_train[i]
        v = np.array(X_raw_counts[i])
        if not lang in lang_dict:
            lang_dict[lang] = v
        else:
            lang_dict[lang] += v

    # to relative
    for lang in lang_dict:
        v = lang_dict[lang]
        lang_dict[lang] = v / np.sum(v)

    return lang_dict


language_dict_unigram = train_lang_dict(X_unigram_train_raw.toarray(), y_train.values)


# Collect relevant chars per language
def getRelevantCharsPerLanguage(features, language_dict, significance=1e-4):
    relevantCharsPerLanguage = {}
   languages = ['French', 'English']
    for lang in languages:
        chars = []
        relevantCharsPerLanguage[lang] = chars
        v = language_dict[lang]
        for i in range(len(v)):
            if v[i] > significance:
                chars.append(features[i])
    return relevantCharsPerLanguage

relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram)
languages=['French','English']

relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram)
languages = ['French', 'English', 'Darija']


# Print number of unigrams per language
for lang in languages:
    print(lang, len(relevantCharsPerLanguage[lang]))

# get most common chars for a few European languages
europeanLanguages = [ 'English', 'French', 'Darija']
relevantChars_OnePercent = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram, 1e-2)

# collect and sort chars
europeanCharacters = []
for lang in europeanLanguages:
    europeanCharacters += relevantChars_OnePercent[lang]
europeanCharacters = list(set(europeanCharacters))
europeanCharacters.sort()

# build data
indices = [unigramFeatures.index(f) for f in europeanCharacters]
data = []
for lang in europeanLanguages:
    data.append(language_dict_unigram[lang][indices])

#build dataframe
df = pd.DataFrame(np.array(data).T, columns=europeanLanguages, index=europeanCharacters)
df.index.name = 'Characters'
df.columns.name = 'Languages'

# plot heatmap
import seaborn as sn
import matplotlib.pyplot as plt
sn.set(font_scale=0.8) # for label size
sn.set(rc={'figure.figsize':(10, 10)})
sn.heatmap(df, cmap="Greens", annot=True, annot_kws={"size": 12}, fmt='.0%')# font size
plt.show()

print(dataset)

