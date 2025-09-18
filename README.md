Twitter Sentiment Analysis using Deep Learning
This project performs sentiment analysis on Twitter data using NLP preprocessing, TF-IDF vectorization, and a Keras deep learning model. It classifies tweets into sentiment categories (e.g., positive, negative, neutral, etc.) based on labeled training data.

# 🔹 Project Workflow
## 1. Importing Libraries
Data Handling: pandas, numpy

Text Preprocessing: re, spacy (en_core_web_sm)

Visualization: matplotlib, seaborn, wordcloud

Modeling: scikit-learn, tensorflow.keras

## 2. Dataset
twitter_training.csv – Training dataset

twitter_validation.csv – Validation dataset

Columns:

id: Unique identifier

src: Source of tweet

label: Sentiment class (Positive / Negative / Neutral / Irrelevant)

tweet: Actual tweet text

## 3. Exploratory Data Analysis (EDA)
Checked dataset size, columns, and missing values.

Visualized label distributions using Seaborn countplot.

Generated WordCloud to understand frequent words.

## 4. Data Preprocessing
Removed null rows in tweet column.

Preprocessed tweets using spaCy:

Tokenization

Lemmatization

Removed stopwords & punctuations

## 5. Vectorization
Applied TF-IDF Vectorization on cleaned tweets.

Split data into train (80%) and validation (20%).

## 6. Label Encoding
Converted categorical sentiment labels into numerical form using LabelEncoder.

## 7. Model Training
Built a feed-forward neural network using Keras:

Input Layer: TF-IDF features

Dense(64, relu)

Dense(32, relu)

Output Layer: softmax (classes = number of unique sentiments)

Loss Function: sparse_categorical_crossentropy

Optimizer: Adam

Metrics: Accuracy

## 8. Model Evaluation
Evaluated on validation and test sets.

Generated:

Classification report (precision, recall, f1-score)

Confusion matrix heatmap

## 9. Testing on Custom Input
You can test the model with your own text:

python
TestingTxt = 'My Friend Pedro is the best'
pre_txt = preprocessing_pipe([TestingTxt])
vect_txt = vect.transform(pre_txt)
pred = model.predict(vect_txt)

testingresult = le.inverse_transform([np.argmax(pred)])
print(testingresult)
# 🔹 Results
Model successfully predicts sentiment classes for Twitter data.

Displays confusion matrix and detailed classification report.

