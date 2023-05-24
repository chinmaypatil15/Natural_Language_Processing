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
df = pd.read_csv(r'G:\project\Natural_Language_Processing\language_detection_data_s.csv',index_col=None)
print(df)

#display the number of texts available  for every class (language)
plt.figure(figsize = (10, 8))
sns.countplot(x=df['Language'])
print(plt.show())

#check missing values
df.isnull().sum()


#Text Cleaning library
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

#Remove english and french stop words
import nltk
stopwords = set(nltk.corpus.stopwords.words('english')) | set(nltk.corpus.stopwords.words('french'))
dataset['Text'] = dataset['Text'].str.lower().apply(lambda x: ' '.join([word for word in str(x).split() if word not in stopwords]))
#delete empty rows
dataset = dataset[dataset['Text']!= '']
#reset data index
dataset=dataset.reset_index().drop('index',axis=1)
print(dataset)

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

from wordcloud import WordCloud
from wordcloud import WordCloud
def plot_wordcloud(docx):
    mywordcloud=WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
#liste of Darija keywords
language_list=dataset['Language'].unique().tolist()
Darija_list=dataset[dataset['Language']=='Darija']['detokenized_sents'].tolist()
Darija_docx=' '.join(Darija_list)
print(plot_wordcloud(Darija_docx))

#liste of English keywords
language_list=dataset['Language'].unique().tolist()
English_list=dataset[dataset['Language']=='English']['detokenized_sents'].tolist()
English_docx=' '.join(English_list)
print(plot_wordcloud(English_docx))

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

# Extract Unigrams
from sklearn.feature_extraction.text import CountVectorizer
unigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
X_unigram_test_raw = unigramVectorizer.transform(X_test)

#getFreatures
unigramFeatures = unigramVectorizer.get_feature_names_out()

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
    languages = ['French', 'English', 'Darija']
    for lang in languages:
        chars = []
        relevantCharsPerLanguage[lang] = chars
        v = language_dict[lang]
        for i in range(len(v)):
            if v[i] > significance:
                chars.append(features[i])
    return relevantCharsPerLanguage

relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram)
languages=['French','English','Darija']
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
print(plt.show())

# number of bigrams
from sklearn.feature_extraction.text import CountVectorizer
bigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))
X_bigram_raw = bigramVectorizer.fit_transform(X_train)
bigramFeatures = bigramVectorizer.get_feature_names_out()
print('Number of bigrams', len(bigramFeatures))

# top bigrams (>1%) for each language
language_dict_bigram = train_lang_dict(X_bigram_raw.toarray(), y_train.values)
relevantCharsPerLanguage = getRelevantCharsPerLanguage(bigramFeatures, language_dict_bigram, significance=1e-2)

# Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
from sklearn.feature_extraction.text import CountVectorizer

top1PrecentMixtureVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), min_df=1e-2)
X_top1Percent_train_raw = top1PrecentMixtureVectorizer.fit_transform(X_train)
X_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(X_test)

language_dict_top1Percent = train_lang_dict(X_top1Percent_train_raw.toarray(), y_train.values)

top1PercentFeatures = top1PrecentMixtureVectorizer.get_feature_names_out()
print('Length of features', len(top1PercentFeatures))
print('')

#Unique features per language
relevantChars_Top1Percent = getRelevantCharsPerLanguage(top1PercentFeatures, language_dict_top1Percent, 1e-5)
for lang in relevantChars_Top1Percent:
    print("{}: {}".format(lang, len(relevantChars_Top1Percent[lang])))


def getRelevantGramsPerLanguage(features, language_dict, top=60):
    relevantGramsPerLanguage = {}
    for lang in languages:
        chars = []
        relevantGramsPerLanguage[lang] = chars
        v = language_dict[lang]
        sortIndex = (-v).argsort()[:top]
        for i in range(len(sortIndex)):
            chars.append(features[sortIndex[i]])
    return relevantGramsPerLanguage


top60PerLanguage_dict = getRelevantGramsPerLanguage(top1PercentFeatures, language_dict_top1Percent)

# top60
allTop60 = []
for lang in top60PerLanguage_dict:
    allTop60 += set(top60PerLanguage_dict[lang])

top60 = list(set(allTop60))

print('All items:', len(allTop60))
print('Unique items:', len(top60))

# getRelevantColumnIndices
def getRelevantColumnIndices(allFeatures, selectedFeatures):
    relevantColumns = []
    for feature in selectedFeatures:
        relevantColumns = np.append(relevantColumns, np.where(allFeatures==feature))
    return relevantColumns.astype(int)

relevantColumnIndices = getRelevantColumnIndices(np.array(top1PercentFeatures), top60)


X_top60_train_raw = np.array(X_top1Percent_train_raw.toarray()[:,relevantColumnIndices])
X_top60_test_raw = X_top1Percent_test_raw.toarray()[:,relevantColumnIndices]

print('train shape', X_top60_train_raw.shape)
print('test shape', X_top60_test_raw.shape)

# Define some functions for our purpose

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import scipy


# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
    print(data_type)
    return None


def normalizeData(train, test):
    train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)
    test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)
    return train_result, test_result


def applyNaiveBayes(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict, clf


def plot_F_Scores(y_test, y_predict):
    f1_micro = f1_score(y_test, y_predict, average='micro')
    f1_macro = f1_score(y_test, y_predict, average='macro')
    f1_weighted = f1_score(y_test, y_predict, average='weighted')
    print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))


def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
    allLabels = list(set(list(y_test) + list(y_predict)))
    allLabels.sort()
    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
    unqiueLabel = np.unique(allLabels)
    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    sn.set(font_scale=0.8)  # for label size
    sn.set(rc={'figure.figsize': (15, 15)})
    sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')  # font size
    plt.show()

# Unigrams
X_unigram_train, X_unigram_test = normalizeData(X_unigram_train_raw, X_unigram_test_raw)
y_predict_nb_unigram,clf1 = applyNaiveBayes(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_nb_unigram)
print(plot_Confusion_Matrix(y_test, y_predict_nb_unigram, "Oranges"))

# Top 1%
X_top1Percent_train, X_top1Percent_test = normalizeData(X_top1Percent_train_raw, X_top1Percent_test_raw)
y_predict_nb_top1Percent,clf2 = applyNaiveBayes(X_top1Percent_train, y_train, X_top1Percent_test)
plot_F_Scores(y_test, y_predict_nb_top1Percent)
print(plot_Confusion_Matrix(y_test, y_predict_nb_top1Percent, "Reds"))

# Top 60
X_top60_train, X_top60_test = normalizeData(X_top60_train_raw, X_top60_test_raw)
y_predict_nb_top60,clf3 = applyNaiveBayes(X_top60_train, y_train, X_top60_test)
plot_F_Scores(y_test, y_predict_nb_top60)
print(plot_Confusion_Matrix(y_test, y_predict_nb_top60, "Greens"))

from sklearn.neighbors import KNeighborsClassifier


def applyNearestNeighbour(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = KNeighborsClassifier()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict, clf


# Unigrams
y_predict_knn_unigram, clf4 = applyNearestNeighbour(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_knn_unigram)
print(plot_Confusion_Matrix(y_test, y_predict_knn_unigram, "Purples"))

error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_unigram_train, y_train)
    pred_i = knn.predict(X_unigram_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title("Taux d'erreurs vs. valeurs de K ")
plt.xlabel('K')
print(plt.ylabel("Taux d'erreurs"))

from sklearn.neighbors import KNeighborsClassifier


def applyNearestNeighbour(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = KNeighborsClassifier(n_neighbors=12)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict, clf


# Unigrams
y_predict_knn_unigram, clf4 = applyNearestNeighbour(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_knn_unigram)
print(plot_Confusion_Matrix(y_test, y_predict_knn_unigram, "Purples"))

# Top 1%
y_predict_knn_top1P,clf5 = applyNearestNeighbour(X_top1Percent_train, y_train,X_top1Percent_test)
plot_F_Scores(y_test, y_predict_knn_top1P)
print(plot_Confusion_Matrix(y_test, y_predict_knn_top1P, "Blues"))

# Top 60
y_predict_knn_top60,clf6 = applyNearestNeighbour(X_top60_train, y_train, X_top60_test)
plot_F_Scores(y_test, y_predict_knn_top60)
print(plot_Confusion_Matrix(y_test, y_predict_knn_top60, "Blues"))

from sklearn.linear_model import LogisticRegression

def applyLogisticRegression(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict, clf


## Unigrams
y_predict_RL_unigram, clf7 = applyLogisticRegression(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_RL_unigram)
plot_Confusion_Matrix(y_test, y_predict_RL_unigram, "Purples")

# Top 1%
y_predict_RL_top1P,clf8 = applyLogisticRegression(X_top1Percent_train, y_train,X_top1Percent_test)
plot_F_Scores(y_test, y_predict_RL_top1P)
plot_Confusion_Matrix(y_test, y_predict_RL_top1P, "Blues")

# Top 60
y_predict_RL_top60,clf9 = applyLogisticRegression(X_top60_train, y_train, X_top60_test)
plot_F_Scores(y_test, y_predict_RL_top60)
plot_Confusion_Matrix(y_test, y_predict_RL_top60, "Blues")

#Tunning param
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=clf8, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_top1Percent_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#k-fold cross validation fct
def ten_fold_cross(model,X_train,y_train):
            cv = KFold(n_splits=10, random_state=1, shuffle=True)
            # create model
            # evaluate model
            scores = cross_val_score(model,  X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
            # report performance
            #print(scores)
            #print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
            #print()
            return np.mean(scores)

#fct to make sure  that the model is well generalized
from sklearn.metrics import accuracy_score
def compare_accuracy_after_and_before_cross(y_test,y_predict,scores):
                    accuracy=accuracy_score(y_test, y_predict)
                    #print('before cross validation, accuracy= ',accuracy)
                    #print()
                    #print('after cross validation, accuracy= ',scores)
#Unigrams NB model
scores=ten_fold_cross(clf1,X_unigram_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_nb_unigram,scores)
#Top 1% NB Model
scores1=ten_fold_cross(clf2,X_top1Percent_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_nb_top1Percent,scores1)
#Top 60 NB Model
scores2=ten_fold_cross(clf3,X_top60_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_nb_top60,scores2)
#Uni-grams KNN  model
scores3=ten_fold_cross(clf4,X_unigram_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_knn_unigram,scores3)
#Top 1% KNN Model
scores4=ten_fold_cross(clf5,X_top1Percent_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_knn_top1P,scores4)
#Top 60 KNN Model
scores5=ten_fold_cross(clf6,X_top60_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_knn_top60,scores5)
#Uni-grams Logistic Regression  model
scores6=ten_fold_cross(clf7,X_unigram_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_RL_unigram,scores6)
#Top 1% Logistic Regression Model
scores7=ten_fold_cross(clf8,X_top1Percent_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_RL_top1P,scores7)
#Top 60 Logistic Regression Model
scores8=ten_fold_cross(clf9,X_top60_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_RL_top60,scores8)


def detect_language(text):
    # vectorize the text
    test = top1PrecentMixtureVectorizer.transform([text])
    var_test = toNumpyArray(test)
    l = clf8.predict(var_test)
    # Check for the prediction probability
    pred_proba = clf8.predict_proba(var_test)
    pred_percentage_for_all = dict(zip(clf8.classes_, pred_proba[0]))
    print("Prediction using Logistic Regression Top 1%:  : {} , Prediction Score : {}".format(l[0], np.max(pred_proba)))
    print()
    print(pred_percentage_for_all)

detect_language('la walakin im not sure that she would be hya')
detect_language('hello world im so happy today')
detect_language('je suis tres heureuse aujourd hui, je me sens tres bien')