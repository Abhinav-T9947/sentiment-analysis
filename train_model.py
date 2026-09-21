import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv(
    "dataset/IMDB_Dataset.csv"
)


print("Dataset Shape:")
print(data.shape)


print("\nFirst 5 Rows:")
print(data.head())


# =========================================================
# 2. CHECK MISSING VALUES
# =========================================================

print("\nMissing Values:")
print(data.isnull().sum())


# =========================================================
# 3. SENTIMENT DISTRIBUTION
# =========================================================

print("\nSentiment Distribution:")
print(data["sentiment"].value_counts())


# =========================================================
# 4. TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML <br> tags
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    # Keep apostrophes
    text = re.sub(
        r"[^a-zA-Z'\s]",
        " ",
        text
    )

    # Normalize contractions
    text = text.replace(
        "didn't",
        "did not"
    )

    text = text.replace(
        "doesn't",
        "does not"
    )

    text = text.replace(
        "don't",
        "do not"
    )

    text = text.replace(
        "wasn't",
        "was not"
    )

    text = text.replace(
        "weren't",
        "were not"
    )

    text = text.replace(
        "isn't",
        "is not"
    )

    text = text.replace(
        "aren't",
        "are not"
    )

    text = text.replace(
        "can't",
        "can not"
    )

    text = text.replace(
        "couldn't",
        "could not"
    )

    text = text.replace(
        "wouldn't",
        "would not"
    )

    text = text.replace(
        "shouldn't",
        "should not"
    )

    text = text.replace(
        "haven't",
        "have not"
    )

    text = text.replace(
        "hasn't",
        "has not"
    )

    text = text.replace(
        "hadn't",
        "had not"
    )

    text = text.replace(
        "never",
        "never"
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning reviews...")

data["review"] = data["review"].apply(
    clean_text
)


# =========================================================
# 5. REMOVE EMPTY REVIEWS
# =========================================================

data = data[
    data["review"].str.strip() != ""
]


# =========================================================
# 6. CONVERT SENTIMENT TO NUMERIC
# =========================================================

data["sentiment"] = data["sentiment"].map(
    {
        "negative": 0,
        "positive": 1
    }
)


# =========================================================
# 7. INPUT AND OUTPUT
# =========================================================

X = data["review"]

y = data["sentiment"]


# =========================================================
# 8. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# =========================================================
# 9. TF-IDF VECTORIZER
# =========================================================

print("\nCreating TF-IDF features...")


tfidf = TfidfVectorizer(

    max_features=30000,

    ngram_range=(1, 2),

    stop_words=None,

    min_df=2,

    max_df=0.95,

    sublinear_tf=True,

    strip_accents="unicode"
)


X_train_tfidf = tfidf.fit_transform(
    X_train
)

X_test_tfidf = tfidf.transform(
    X_test
)


print(
    "\nTF-IDF Training Shape:"
)

print(
    X_train_tfidf.shape
)


print(
    "\nTF-IDF Testing Shape:"
)

print(
    X_test_tfidf.shape
)


# =========================================================
# 10. NAIVE BAYES
# =========================================================

print("\n" + "=" * 60)

print("NAIVE BAYES")

print("=" * 60)


nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)


nb_prediction = nb_model.predict(
    X_test_tfidf
)


nb_accuracy = accuracy_score(
    y_test,
    nb_prediction
)

nb_precision = precision_score(
    y_test,
    nb_prediction
)

nb_recall = recall_score(
    y_test,
    nb_prediction
)

nb_f1 = f1_score(
    y_test,
    nb_prediction
)


print(
    "Accuracy :",
    round(nb_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(nb_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(nb_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(nb_f1 * 100, 2),
    "%"
)


# =========================================================
# 11. LOGISTIC REGRESSION
# =========================================================

print("\n" + "=" * 60)

print("LOGISTIC REGRESSION")

print("=" * 60)


logistic_model = LogisticRegression(

    max_iter=1000,

    C=2.0,

    solver="liblinear",

    random_state=42
)


logistic_model.fit(
    X_train_tfidf,
    y_train
)


logistic_prediction = logistic_model.predict(
    X_test_tfidf
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

logistic_precision = precision_score(
    y_test,
    logistic_prediction
)

logistic_recall = recall_score(
    y_test,
    logistic_prediction
)

logistic_f1 = f1_score(
    y_test,
    logistic_prediction
)


print(
    "Accuracy :",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(logistic_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(logistic_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(logistic_f1 * 100, 2),
    "%"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_prediction
    )
)


# =========================================================
# 12. LINEAR SVM
# =========================================================

print("\n" + "=" * 60)

print("LINEAR SVM")

print("=" * 60)


svm_model = LinearSVC(
    random_state=42
)


svm_model.fit(
    X_train_tfidf,
    y_train
)


svm_prediction = svm_model.predict(
    X_test_tfidf
)


svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)

svm_precision = precision_score(
    y_test,
    svm_prediction
)

svm_recall = recall_score(
    y_test,
    svm_prediction
)

svm_f1 = f1_score(
    y_test,
    svm_prediction
)


print(
    "Accuracy :",
    round(svm_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(svm_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(svm_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(svm_f1 * 100, 2),
    "%"
)


# =========================================================
# 13. SELECT FINAL MODEL
# =========================================================

print("\n" + "=" * 60)

print("FINAL MODEL")

print("=" * 60)


# Logistic Regression is selected because
# the Streamlit application needs probabilities.

final_model = logistic_model


print(
    "Selected Model: Logistic Regression"
)

print(
    "Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)


# =========================================================
# 14. SAVE MODEL
# =========================================================

print("\nSaving model...")


joblib.dump(
    final_model,
    "model/sentiment_model.pkl"
)


joblib.dump(
    tfidf,
    "model/tfidf_vectorizer.pkl"
)


print(
    "Model saved successfully!"
)


# =========================================================
# 15. TEST MODEL
# =========================================================

print("\n" + "=" * 60)

print("TESTING MODEL")

print("=" * 60)


test_reviews = [

    "The movie was absolutely fantastic",

    "The movie was amazing",

    "The movie was very good",

    "I really enjoyed this movie",

    "The movie was not good",

    "The movie was not bad",

    "The movie was terrible",

    "The movie was extremely boring",

    "I did not enjoy this movie",

    "I would definitely recommend this movie",

    "This movie was not enjoyable",

    "The story was not interesting",

    "The acting was not good",

    "I never liked this movie"
]


for review in test_reviews:

    cleaned_review = clean_text(
        review
    )


    review_vector = tfidf.transform(
        [cleaned_review]
    )


    prediction = final_model.predict(
        review_vector
    )[0]


    probabilities = final_model.predict_proba(
        review_vector
    )[0]


    negative_probability = (
        probabilities[0] * 100
    )


    positive_probability = (
        probabilities[1] * 100
    )


    sentiment = (
        "POSITIVE"
        if prediction == 1
        else "NEGATIVE"
    )


    print("\nReview:", review)

    print(
        "Prediction:",
        sentiment
    )

    print(
        f"Negative: {negative_probability:.2f} %"
    )

    print(
        f"Positive: {positive_probability:.2f} %"
    )


# =========================================================
# 16. TRAINING COMPLETE
# =========================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)
