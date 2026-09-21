# 🎬 Movie Review Sentiment Analysis

A machine learning web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**.

The project uses **Natural Language Processing (NLP)**, **TF-IDF**, and **Logistic Regression**, with an interactive **Streamlit** interface.

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing technique used to identify the emotional tone of text.

This project uses the **IMDb 50,000 Movie Reviews dataset** to train a machine learning model that classifies movie reviews into two categories:

* 😊 Positive
* 😞 Negative

The trained model is integrated into a Streamlit web application where users can enter reviews and receive predictions along with confidence and sentiment probabilities.

## 🎯 Objectives

* Build a machine learning model for movie review sentiment classification.
* Apply NLP techniques to process textual data.
* Convert text into numerical features using TF-IDF.
* Compare multiple machine learning algorithms.
* Develop an interactive web application using Streamlit.
* Provide sentiment prediction with confidence information.
* Add live autocomplete and sentence suggestions to improve the user experience.

## 📊 Dataset

**Dataset:** IMDb 50K Movie Reviews

The dataset contains:

* **50,000 movie reviews**
* **25,000 positive reviews**
* **25,000 negative reviews**

The data contains two main columns:

| Column      | Description          |
| ----------- | -------------------- |
| `review`    | Movie review text    |
| `sentiment` | Positive or Negative |

The dataset is not included in this repository.

## ⚙️ Machine Learning Workflow

The project follows this workflow:

```text
IMDb Movie Reviews
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Logistic Regression
        ↓
Save Trained Model
        ↓
Streamlit Web Application
```

## 🧹 Text Preprocessing

The review text is cleaned before being passed to the machine learning model.

The preprocessing includes:

* Converting text to lowercase
* Removing HTML `<br>` tags
* Normalizing common contractions
* Removing unnecessary characters
* Removing extra spaces

## 🔤 TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert movie reviews into numerical feature vectors.

The model uses:

* Up to **30,000 features**
* Unigrams and bigrams
* Minimum document frequency filtering
* Maximum document frequency filtering
* Sublinear TF scaling

## 🤖 Machine Learning Models

Three classification algorithms were evaluated:

| Model               |   Accuracy |
| ------------------- | ---------: |
| Naive Bayes         |     87.95% |
| Logistic Regression | **91.44%** |
| Linear SVM          |     90.92% |

### Final Model

**Logistic Regression** was selected as the final model.

Test accuracy:

**91.44%**

The Logistic Regression model was selected for the application because it provides class probabilities that can be displayed as prediction confidence and sentiment probabilities.

## 🌐 Streamlit Application

The project includes an interactive Streamlit web interface.

### Features

* 📝 Movie review input
* ✨ Live autocomplete suggestions
* 💬 Sentence suggestions
* 🖱️ Click-to-select suggestions
* ⌨️ Keyboard navigation
* 🔍 Analyze Review button
* 😊 Positive sentiment prediction
* 😞 Negative sentiment prediction
* 📊 Confidence percentage
* 📈 Positive/negative probability
* ⚠️ Ambiguous review detection
* ⚠️ Unknown-word handling

### Application Workflow

```text
Enter Review
     ↓
Autocomplete Suggestions
     ↓
Click / Edit Suggestion
     ↓
Analyze Review
     ↓
Text Preprocessing
     ↓
TF-IDF Transformation
     ↓
Logistic Regression
     ↓
Sentiment + Confidence
```

## 📁 Project Structure

```text
sentiment-analysis/
│
├── autocomplete_component/
│   ├── __init__.py
│   └── frontend/
│       └── index.html
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── train_model.py
├── test_component.py
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* Streamlit
* HTML
* CSS
* JavaScript
* Natural Language Processing (NLP)

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Abhinav-T9947/sentiment-analysis.git
```

### 2. Open the project folder

```bash
cd sentiment-analysis
```

### 3. Install the required packages

```bash
pip install pandas scikit-learn joblib streamlit
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Example

### Positive Review

```text
The movie was absolutely fantastic
```

Example output:

```text
😊 POSITIVE
Confidence: 74.00%
```

### Negative Review

```text
The movie was extremely boring
```

The application predicts the corresponding negative sentiment and displays the model confidence.

## 📈 Model Performance

The final Logistic Regression model achieved:

* **Accuracy:** 91.44%
* **Precision:** 90.92%
* **Recall:** 92.08%
* **F1-Score:** 91.49%

These results were obtained on the held-out IMDb test data.

## ⚠️ Limitations

The model predicts only two sentiment classes:

* Positive
* Negative

Neutral or mixed reviews may be difficult to classify because the model was trained primarily on positive and negative examples.

Model confidence is an indication of the model's prediction probability and should not be interpreted as guaranteed correctness.

## 🔮 Future Scope

Possible future improvements include:

* Support for neutral sentiment
* Improved handling of negation
* More advanced NLP techniques
* Transformer-based models such as BERT
* Multilingual sentiment analysis
* Deployment to a public cloud platform
* Larger and more diverse datasets
* More advanced spell correction and autocomplete

## 👨‍💻 Author

**Abhinav T**

Built as a machine learning and NLP project using Python and Streamlit.
