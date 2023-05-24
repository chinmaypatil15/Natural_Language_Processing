import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv(r"Language_Detection.csv")
df.head(10)
print(df)
print(df.shape)
print(df.info())
df.isnull().sum()
print(df[df.duplicated()])
print(len(df[df.duplicated()]))
print(df.drop(df[df.duplicated()].index, axis=0, inplace=True))
print(df.shape)
print(df["Language"].nunique())
print(df["Language"].value_counts())

plt.figure(figsize=(20, 8))

total = float(len(df['Language']))
ax = sns.countplot(x='Language', data=df, order=df['Language'].value_counts().index, palette='magma')

for p in ax.patches:
    percentage = '{:.2f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), fontsize=16, ha='center')

plt.title('Counts and Percentages of Languages', fontsize=24)
plt.xlabel("Language", fontsize=20)
plt.ylabel("Count", fontsize=20)
plt.xticks(size=18, rotation=90)
print(plt.show())

language= df['Language'].value_counts().reset_index()
print(language)

plt.figure(figsize=(10,10))

#create pie chart
labels= language['index']
plt.pie(language["Language"], labels= labels, autopct='%.1f%%', textprops={'fontsize': 15})
print(plt.show())

df1= df.copy()
df1['cleaned_Text']= ""
print(df1)

import re


def clean_function(Text):
    # removing the symbols and numbers
    Text = re.sub(r'[\([{})\]!@#$,"%^*?:;~`0-9]', ' ', Text)

    # converting the text to lower case
    Text = Text.lower()
    Text = re.sub('http\S+\s*', ' ', Text)  # remove URLs
    Text = re.sub('RT|cc', ' ', Text)  # remove RT and cc
    Text = re.sub('#\S+', '', Text)  # remove hashtags
    Text = re.sub('@\S+', '  ', Text)  # remove mentions
    Text = re.sub('\s+', ' ', Text)  # remove extra whitespace

    return Text

df1['cleaned_Text'] = df1['Text'].apply(lambda x: clean_function(x))
print(df1)

X= df1["cleaned_Text"]
y= df1["Language"]

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
y= encoder.fit_transform(y)

from sklearn.feature_extraction.text import CountVectorizer
CV= CountVectorizer()
X= CV.fit_transform(X).toarray()
print(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

models = {
    'K-Nearest Neighbors' : KNeighborsClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'MNB' : MultinomialNB()
}

#%%time
for name, model in models.items():
    print(f'{name} training started...')
    model.fit(X_train, y_train)
    print(f'{name} trained')

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report

#%%time
for name in models:
    acc_score= round(accuracy_score(y_test, models.get(name).predict(X_test)), 3)
    print(f'{name} accuracy score :  {acc_score}')

for name in models:
    print(f'{name} classification report')
    print("-------------------------------")
    print(classification_report(y_test, models.get(name).predict(X_test)))
    print("******************************")
    print(" ")

for name in models:
    print(f'{name} ConfusionMatrix')
    predictions= models.get(name).predict(X_test)
    score = round(accuracy_score(y_test, models.get(name).predict(X_test)), 3)
    confusionMatrix = CM(y_test, models.get(name).predict(X_test))
    sns.heatmap(confusionMatrix, annot=True, fmt=".0f")
    print(plt.xlabel('Actual Values'))
    print(plt.ylabel('Prediction Values'))
    print(plt.title('Accuracy Score: {0}'.format(score), size = 15))
    print(plt.show())
    print("******************************")
    print(" ")

def prediction(text):
    x= CV.transform([text]).toarray()
    lang= model.predict(x)
    lang= encoder.inverse_transform(lang)
    print("This word/sentence contains {} word(s).".format(lang[0]))


print(prediction("Your memory improves as you learn a language. In addition, since your brain will automatically translate, it enables the brain to work in a versatile way and contributes to the development of your abilities."))
print(prediction("L'apprentissage d'une langue améliore la mémoire. De plus, comme votre cerveau traduira automatiquement, cela lui permet de travailler de manière polyvalente et contribue au développement de vos compétences."))
print(prediction("Η μνήμη σας βελτιώνεται καθώς μαθαίνετε μια γλώσσα. Επιπλέον, δεδομένου ότι ο εγκέφαλός σας θα μεταφραστεί αυτόματα, δίνει τη δυνατότητα στον εγκέφαλο να λειτουργεί με ευέλικτο τρόπο και συμβάλλει στην ανάπτυξη των ικανοτήτων σας."))
