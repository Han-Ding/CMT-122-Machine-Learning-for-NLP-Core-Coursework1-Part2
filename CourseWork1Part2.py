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
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.sparse import hstack, csr_matrix

# First read the target text and select the first 10,000 lines to avoid processing less than all the data.
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))
    print(result)

# Here the code is used to automatically detect the encoding type of the target file.
try:
    encoding = result['encoding']
    df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', encoding=encoding)
except UnicodeDecodeError:
    # If the auto-detected encoding still reports an error, it will be skipped, try to apply latin1, and if there is still an error, it will be skipped directly.
    df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', encoding='latin1', errors='replace')

# Separate text data in preparation for processing labels and preprocessing.
X = df['text']
y = df['category']

# In order to become a machine-recognisable language, converting categorical labels to numerical labels.
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

X_processed = X.apply(preprocess_text)

# # Feature extraction
# # Feature 1: Word Frequency Feature (TF-IDF)
# tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# X_tfidf = tfidf_vectorizer.fit_transform(X_processed)

# # Feature 2: Bag-of-words model feature (CountVectorizer)
# count_vectorizer = CountVectorizer(max_features=5000)
# X_count = count_vectorizer.fit_transform(X_processed)

# In addition to calling python library functions, it is also possible to calculate TF-IDF manually.
def custom_tfidf(corpus, max_features=5000):
    word_counts = [Counter(text.split()) for text in corpus]
    N = len(corpus)
    df = Counter()

    for wc in word_counts:
        for word in wc:
            df[word] += 1

    idf = {word: np.log(N / (df[word] + 1)) for word in df}
    tfidf_vectors = []

    for doc in word_counts:
        tfidf = {}
        for word, count in doc.items():
            tf = count / len(doc)
            tfidf[word] = tf * idf[word]
        tfidf_vectors.append(tfidf)
        
    vocabulary = list(idf.keys())
    vocabulary.sort()
    
    X_tfidf = np.zeros((len(corpus), len(vocabulary)))
    for i, tfidf in enumerate(tfidf_vectors):
        for word, value in tfidf.items():
            if word in vocabulary:
                X_tfidf[i, vocabulary.index(word)] = value
    
    if max_features is not None:
        feature_sums = X_tfidf.sum(axis=0)
        sorted_idx = feature_sums.argsort()[::-1][:max_features]
        X_tfidf = X_tfidf[:, sorted_idx]
        vocabulary = [vocabulary[i] for i in sorted_idx]
    
    return X_tfidf, vocabulary

X_tfidf, tfidf_vocab = custom_tfidf(X_processed.values.tolist(), max_features=5000)

# Similarly bag-of-words models can be computed manually.
def custom_bow(corpus, max_features=5000):
    word_counts = [Counter(text.split()) for text in corpus]
    
    vocabulary = list(set([word for wc in word_counts for word in wc]))
    vocabulary.sort()
    
    X_count = np.zeros((len(corpus), len(vocabulary)))
    for i, wc in enumerate(word_counts):
        for word, count in wc.items():
            if word in vocabulary:
                X_count[i, vocabulary.index(word)] = count
    
    if max_features is not None:
        feature_sums = X_count.sum(axis=0)
        sorted_idx = feature_sums.argsort()[::-1][:max_features]
        X_count = X_count[:, sorted_idx]
        vocabulary = [vocabulary[i] for i in sorted_idx]
    
    return X_count, vocabulary

X_count, bow_vocab = custom_bow(X_processed.values.tolist(), max_features=5000)


# Feature 3: Article Length Feature
article_length = X_processed.apply(lambda x: len(x.split()))
article_length = article_length.values.reshape(-1, 1)

# Feature 4: Average word length
avg_word_length = X_processed.apply(lambda x: np.mean([len(word) for word in x.split()]))
avg_word_length = avg_word_length.values.reshape(-1, 1)

# Feature 5: Number of special symbols
special_char_count = X.apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)))
special_char_count = special_char_count.values.reshape(-1, 1)

# hstack() is suitable for merging sparse matrices (or arrays of the same shape) with the same number of rows. The currently provided feature matrices do not have matching shapes to complete the merge operation. So it is necessary to convert the last three features into the form of sparse matrices as well.
article_length = csr_matrix(article_length)
avg_word_length = csr_matrix(avg_word_length)
special_char_count = csr_matrix(special_char_count)

# Combine all features
X_features = hstack([X_tfidf, X_count, article_length, avg_word_length, special_char_count])

# The text was first divided into training and test sets in 7:3, and then the test set was divided into development and test sets in 1:1.
X_train, X_temp, y_train, y_temp = train_test_split(X_features, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optimise the number of feature selections on the development set
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

# Cross-validation
cv_scores = cross_val_score(classifier, X_selected, y, cv=5)
print(f"Average accuracy of cross-validation: {np.mean(cv_scores)}")

# Confusion matrix (math.)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# SVM model training
svm_classifier = SVC(kernel='linear',max_iter=7000, C=1.0)
svm_classifier.fit(X_train, y_train)

# Prediction using SVM
y_pred_svm = svm_classifier.predict(X_test)

# Evaluation
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svm, annot=True, cmap='Reds', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()