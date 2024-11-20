## _**Goal**_
**In Part 2, you are provided with a text classification dataset (named “bbc-text”). The 
dataset contains news articles assigned to five categories: tech, business, sport, politics 
and entertainment. Using this dataset, you need to preprocess the data, select features, 
train and evaluate a machine learning model of your choice to classify the news articles. 
You should include at least three different features to train your model, one of them 
should be based on some sort of word frequency.  
You can decide on the type of frequency (absolute or relative, normalised or not). Text 
preprocessing is mandatory for the word frequency feature.  
The remaining two (or more) features can be chosen freely. Then, you will have to 
perform feature selection to reduce the dimensionality of all the features.**

---

# _Main idea:_
 1. Import necessary libraries: we use pandas for data processing and scikit-learn for feature extraction, selection and model training.
 2. Load dataset: Suppose the dataset contains ‘test’ and ‘category’ columns.
 3. Data preprocessing: convert all article texts to lowercase, remove punctuation, stop words and numbers, and then split words.
 4. Feature extraction:
 - Feature 1: extract word frequency features using TF-IDF.
 - Feature 2: Extract word frequency features using bag-of-words model.
 - Feature 3: Length features of articles.
 - Feature 4: Average length of words.
 - Feature 5: Number of special symbols.
 5. Feature Selection: Select the best features using chi-square test to reduce the number of features and prevent overfitting.
 6. Dataset Splitting: Split the dataset into a training set, a test set adn a development set, with the test set accounting for 15%.
 7. Train the logistic regression model and SVM model then make predictions on the test set.
 8. Evaluate model performance: evaluate the model using classification reports and confusion matrices, and check the stability of the model using cross-validation.

---

# **Note: _To use this code, simply change the absolute path to the file bbc-text.csv._**

---

# _First Step: Introducing the required libraries_
**Note: _Without going into too much detail here, the main point is to show some of the new libraries compared to Part1._**

- **chardet**: Used to detect the encoding format of CSV files so that files containing special characters can be read correctly.

- **cross_val_score**: Used for cross-validation to assess model stability and performance.

- **TfidfVectorizer**: Used to convert text into TF-IDF feature vectors to measure the importance of words.

- **LabelEncoder**: Used to convert category labels to numeric labels for processing by the model.

- **chi2**: A chi-square test method for selecting the features with the strongest correlation with the category labels.

---

# _Second Step: Load dataset_
- **Detecting file encoding**

```
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))
    print(result)
```

- **Read the file using the detected encoding**

```
try:
    encoding = result['encoding']
    df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', encoding=encoding)
except UnicodeDecodeError:
    # If the auto-detected encoding still reports an error, you can try a common encoding or ignore the error
    df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\bbc-text.csv', encoding='latin1', errors='replace')
```
**Note: _Here and Part1 in the processing is basically the same, but in accordance with the detection of text encoding type processing text will encounter new problems, 
some special characters can not be recognised, will still be reported as an error, here to take some other common encoding ‘latin1’ to try, 
and if still can not be recognised then If the character is still not recognised, it will be replaced by a special character. <This method affects the integrity of the text and may affect the final accuracy, 
but I haven't found a better solution yet>_**

- **Separate feature and target columns in the dataset for the following vectorisation**

```
X = df['text']
y = df['category']
```

- **Converting categorical labels to numerical labels**

```
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

**Note: _The purpose of this step is to convert text labels into numeric values, since the model cannot handle characters. 
Here I plan to process all five feature values, using fit_transform() to transform all the labels (‘business’, ‘sport’, ‘politics’, ‘technology’, ‘entertainment’) into (0, 1, 2, 3, 4)._**

---

## _**Third Step: Text Preprocessing Functions**._

```
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
```

**Note: _This part is also much the same as Part 1, so I won't explain too much._**

---

## _**Forth Step: Feature extraction**._
**Note: _This step is the most important part of the entire code. When faced with data that has multiple eigenvalues, then the linear regression model will no longer work. 
The following will break down each of the 5 feature choices in classification models and why it was chosen and its advantages and disadvantages._**

- **Feature 1: Word Frequency Feature (TF-IDF)**

```
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X_processed)
```
**Note: _First of all TfidfVectorizer is a class in scikit-learn, located in the sklearn.feature_extraction.text module.
It is fitting and transforming the dataset X_processed, calculating the TF-IDF weights and converting the text into a TF-IDF feature matrix.
Only 5000 features will be retained in the end (you can change this yourself, try to keep it modest, not too small and not too much)._**

**Average: Effective capture is possible for common words**

**Neg: Then the drawbacks are obvious, the diversity of words in this method is not sensitive, because it does not capture the context**

---

- **Feature 2: Bag-of-words model feature (CountVectorizer)**

```
count_vectorizer = CountVectorizer(max_features=5000)
X_count = count_vectorizer.fit_transform(X_processed)
```

**Note: _The basic idea of the bag-of-words model is to ignore the word order and grammatical structure of the text and instead simply count the number of times each word appears in the text.
Words are first extracted from all the text and a unique index is assigned to each word. For each document, the number of occurrences of each word in the vocabulary is counted and represented as a feature vector. 
This simply means processing the text data into machine understandable numerical data._**

**Average: No complex mathematical calculations are required. By counting word frequencies, the text can be converted into numerical vectors.**

**Neg: As with TF-IDF, context is ignored and computational complexity increases for large-scale documents.**

---

- **Feature 3: Article Length Feature**

```
article_length = X_processed.apply(lambda x: len(x.split()))
article_length = article_length.values.reshape(-1, 1)
```

**Note: _The article length feature simply counts the number of words in each article and converts it into a two-dimensional array that records the length of each sentence._**

**Average: A little bit obvious is that the statistics is simple, reducing the computational complexity, and for some specific features classification will have better results, 
for example, news articles will generally be longer than the entertainment category, the technology category also has a longer description.**

**Neg: The downside is that it is very limiting and is judged solely by virtue of the length of the article.**

---

- **Feature 4: Average word length**

```
avg_word_length = X_processed.apply(lambda x: np.mean([len(word) for word in x.split()]))
avg_word_length = avg_word_length.values.reshape(-1, 1)
```

**Note: _Word average length is also well understood, is to calculate the length of all words in each text and finally divided by the number of words derived from the results and the text into a two-dimensional array._**

**Average: The advantages are basically the same as the method for article length characterisation..**

**Neg: Provides less information to capture the specifics and context of the text.**

---

- **Feature 5: Number of special symbols**

```
special_char_count = X.apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)))
special_char_count = special_char_count.values.reshape(-1, 1)
```

**Note: _Count the number of all special symbols in the article.
This feature is mainly used to reflect how many special symbols are used in the text as a way to judge the style and emotional intensity of the text._**

**Average: There are better recognition results for texts with strong emotional tendencies.**

**Neg: Limited information, constrained by the volume of the text. There is also the possibility of overlapping categories directly.**

- **Merge all features**
```
from scipy.sparse import hstack
X_features = hstack([X_tfidf, X_count, article_length, avg_word_length, special_char_count])

```
**Note: _Finally the hstack() method is used to combine the different features to form a large feature matrix._**

---

## _**Fifth Step: Optimising the number of feature choices on a development set**._
**Note: _This step is also the most confusing point for me at the beginning, because using the experience of the previous Part1 to see that only need to carry out several iterations to make the data reach an optimal solution of its own, 
but in the actual debugging I found that no matter how I adjusted the size of iter, there will always be an error report. 
Finally I found a specific solution, because unlike the Part1 part, this time there are multiple eigenvalues, so the number of iterations should be very large in order to satisfy, 
but if each time to manually debugging and will be an additional waste of time, so I wondered whether it can and its own to find an optimal solution._**

```
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
```
**Note: _The main purpose of this code is to set some iter values and compare the iter with the highest accuracy.
The idea of the finding process is similar to the bubble sort in c, where the larger number is selected each time and the value and subscript are kept at the end._**

- **Reselect features using the optimal number of features**
```
selector = SelectKBest(chi2, k=best_k)
X_selected = selector.fit_transform(X_features, y)
```

---

## _**Sixth Step: Training Model**._

- **The original data be separated into three distinct sets: one for training, one for testing and one for val**

```
X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
**_Note: Compared to Part1 here is another development set added. The reason for choosing him is that the development set has a tuning function for the model, 
which improves the generalisation ability of the model and also prevents overfitting, so that the model performs well with new data.
The division is done by dividing the raw text into training set and others in the ratio of 7:3 and then dividing the others into development set and test set in the ratio of 1:1.
The final ratio of training set, development set and test set is 70%,15%,15%._**

- **Training logistic regression models and SVM Model**

```
classifier = LogisticRegression(solver='saga', max_iter=7000, C=0.1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```
**_Note: Here, max_iter is also chosen to be 7000 because of the increase in the number of eigenvalues, and the value of c is adjusted downwards because c is the inverse of regularisation, 
so the smaller the value, the stronger the regularisation and the lesser the likelihood of overfitting the model._**

- **Print Report**

```
print("Classification report: ")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

**_Note: In contrast to Part1, a lazy library is used here, classification_report(), whose role is to be used specifically to generate a detailed report on Precision, Recall, F1-Scocre and Support._**

- **SVM model training**

```
svm_classifier = SVC(kernel='linear',max_iter=7000, C=1.0)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
```
**_Note: The binary SVM model added here is also a classification model and its purpose is to compare the training results of the two models._**

---

## _**Seventh Step: Print Confusion Matrix**._

```
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
```
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svm, annot=True, cmap='Reds', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()
```

**_Note: Same function as Part1 and will not go into too much detail._**




---

## **_Conclusion:_**

**_Compared to Part1, Part2 explores more about how text data with multiple feature values should be processed. Starting from the eigenvalue division, the linear regression model is no longer applicable, 
so choose a classification model that performs better. After that, the text is divided into three, training set, test set and development set is to better adapt the model to new data and avoid overfitting phenomenon. 
At the same time, the phenomenon of multiple feature values will also need to increase the number of iterations is the model to reach the optimal solution. There are still deficiencies in the code and the need to modify, 
but also need to continue to learn._**

## **_Critical Reflection:_**

**_1. In terms of model selection, I just used a classification model based on logistic regression, but I'm not sure if it's optimal or not, there are other more sophisticated models, for example,
I've used a plain Bayesian model before to process text with over 4w data for sentiment analysis. But the accuracy was only about 85%, so I think it might be the amount of data that makes this not-so-accurate model as accurate as 95%._**

**_2.In the model i just set some parameters manually like c=0.1, max_iter=7000, etc. These parameters are not always the most suitable, so I think I should let the machine simulate the exercise itself to find out the most suitable parameters._**

**_3.This model also does not do a good job of dealing with the context of the text, such as contextual scenarios and the like.
And the data according to the source is slightly single, which also affects the accuracy of the model._**




