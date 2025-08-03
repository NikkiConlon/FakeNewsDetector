# Fake News Detector

This project trains a logistic regression model to classify news articles as **fake** or **real** based on their text content.

## Setup

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Place the `Fake.csv` and `True.csv` files in the project root directory.

3. Run the combine script to merge these datasets into a single file:

    ```bash
    python combine_news.py
    ```

4. Train and evaluate the model:

    ```bash
    python fake_news_detector.py
    ```

The trained model and vectorizer will be saved as `fake_news_model.pkl` and `tfidf_vectorizer.pkl` respectively.

## Usage

Once the model is trained and saved, you can load it to predict new texts without retraining:

```python
import joblib

clf = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict(text):
    X = vectorizer.transform([text])
    pred = clf.predict(X)
    return "REAL" if pred[0] == 1 else "FAKE"

# Exa
