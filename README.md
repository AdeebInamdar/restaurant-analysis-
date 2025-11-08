# 💬 Sentiment Analysis of Restaurant Reviews (NLP Project)

## Project Overview

This repository contains a foundational Natural Language Processing (NLP) project focused on **Sentiment Analysis**. The goal is to classify restaurant reviews as either positive ("Liked") or negative ("Not Liked") using a Machine Learning approach.

The project demonstrates a complete NLP workflow, from raw text ingestion to final model evaluation.

## Methodology

The analysis follows these key steps:

1.  **Data Ingestion:** Reading and inspecting the `Restaurant_Reviews.tsv` dataset.
2.  **Text Preprocessing:** A crucial step where text is cleaned by:
    * Removing punctuation and numbers.
    * Converting text to lowercase.
    * Applying **Stemming** (e.g., reducing "loved" to "love").
    * Performing **Stopword Removal** (while strategically keeping words like "not" for sentiment accuracy).
3.  **Feature Extraction:** Converting the cleaned text into a numerical format using the **Bag-of-Words (BoW)** model and **CountVectorizer** (limiting to the top 1500 features).
4.  **Model Training:** Training a **Gaussian Naive Bayes** classifier, a highly effective and common model for text classification.
5.  **Evaluation:** Assessing model performance using a **Confusion Matrix** and calculating the final **Accuracy Score** on the test set.

## Technical Stack

* **Language:** Python
* **Libraries:**
    * `pandas`, `numpy` (Data handling)
    * `nltk` (Natural Language Toolkit for stemming and stopwords)
    * `re` (Regular Expressions for cleaning)
    * `sklearn` (Scikit-learn for `CountVectorizer`, `GaussianNB`, `confusion_matrix`, and `accuracy_score`)
* **Environment:** Jupyter Notebook (`.ipynb`)

## Results

The Naive Bayes model achieved an accuracy of **73%** in correctly predicting the sentiment of the test reviews. The confusion matrix provides a detailed breakdown of true and false predictions.

## How to Run

1.  **Clone this repository:** `git clone [Your Repository URL]`
2.  **Ensure you have Python and necessary libraries:**
    ```bash
    pip install pandas numpy nltk scikit-learn
    ```
3.  **Open the Notebook:** Run the `Natural_language_processing.ipynb` file in a Jupyter environment (Jupyter Lab, VS Code, or Google Colab).
