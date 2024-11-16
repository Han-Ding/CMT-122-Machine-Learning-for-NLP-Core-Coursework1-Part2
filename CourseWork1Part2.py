import nltk
import chardet
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from collections import Counter

# Detecting file encoding
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))
    print(result)

# Read the file using the detected encoding
try:
    encoding = result['encoding']
    df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', encoding=encoding)
except UnicodeDecodeError:
    # If the auto-detected encoding still reports an error, you can try a common encoding or ignore the error
    df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', encoding='latin1', errors='replace')

# The dataset has two columns: ‘text’ and ‘category’.
X = df['text']
y = df['category']

# Converting categorical labels to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Text Preprocessing Functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip out the numbers
    text = re.sub(r'\d+', '', text)
    # Participle
    words = text.split()
    # Disjunction
    stopwords_set = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwords_set]
    return ' '.join(words)

# Pre-processing of all articles
X_processed = X.apply(preprocess_text)

# Feature extraction
# Feature 1: Word Frequency Feature (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X_processed)

# Feature 2: Bag-of-words model feature (CountVectorizer)
count_vectorizer = CountVectorizer(max_features=5000)
X_count = count_vectorizer.fit_transform(X_processed)

# Feature 3: Article Length Feature
article_length = X_processed.apply(lambda x: len(x.split()))
article_length = article_length.values.reshape(-1, 1)

# Feature 4: Average word length
avg_word_length = X_processed.apply(lambda x: np.mean([len(word) for word in x.split()]))
avg_word_length = avg_word_length.values.reshape(-1, 1)

# Feature 5: Number of special symbols
special_char_count = X.apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)))
special_char_count = special_char_count.values.reshape(-1, 1)

# Combine all features
from scipy.sparse import hstack
X_features = hstack([X_tfidf, X_count, article_length, avg_word_length, special_char_count])

# The dataset is split into a training set, a development set and a test set
X_train, X_temp, y_train, y_temp = train_test_split(X_features, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optimise the number of feature selections on the development set
from sklearn.metrics import accuracy_score
best_k = 0
best_score = 0
for k in [1000, 2000, 3000, 4000]:
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    classifier = LogisticRegression(solver='saga', max_iter=5000, C=0.1)
    classifier.fit(X_train_selected, y_train)
    y_val_pred = classifier.predict(X_val_selected)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"k={k}, Validation Accuracy: {val_accuracy}")
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_k = k
print(f"The optimal number of features is: {best_k}, The validation set accuracy is: {best_score}")

# Reselect features using the optimal number of features
selector = SelectKBest(chi2, k=best_k)
X_selected = selector.fit_transform(X_features, y)

# The dataset after using the final feature selection is split into training, development and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Training logistic regression models
classifier = LogisticRegression(solver='saga', max_iter=7000, C=0.1)
classifier.fit(X_train, y_train)

# Prediction using classification models
y_pred = classifier.predict(X_test)

# Assessment model
print("Classification report: ")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix (math.)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation
cv_scores = cross_val_score(classifier, X_selected, y, cv=5)
print(f"Average accuracy of cross-validation: {np.mean(cv_scores)}")

