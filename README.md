# Fake News Detector

This project uses a logistic regression model to classify news articles as fake or real based on their text content. It contains data preprocessing, model training, evaluation, and prediction.

## Features

- Implements TF-IDF vectorization to convert textual data into meaningful numerical representations for model input.

- Utilizes Logistic Regression, a robust and interpretable classification algorithm, to differentiate between fake and real news articles.

- Provides comprehensive evaluation metrics including precision, recall, and F1-score for rigorous assessment of classification performance.

- Includes detailed visualizations, such as confusion matrices and classification reports, to enable thorough analysis of model results.

- Designed for extensibility, facilitating adaptation to alternative datasets or integration of more advanced modeling techniques.

- Supports prediction on preprocessed datasets with scope for future enhancements to handle novel text inputs.

## How to Run

### 1. Clone Repository and Set Up
- Clone the repository
git clone: https://github.com/NikkiConlon/FakeNewsDetector.git
      
- Navigate into the project directory:

      cd FakeNewsDetector

- Create and activate a virtual environment:

        python -m venv .venv
  # On Windows:
      .venv\Scripts\activate
  # On macOS/Linux:
      source .venv/bin/activate

- Install dependencies:

        pip install -r requirements.txt

### 2. Download and Prepare Data 
- Download the datasets (Fake.csv and True.csv) from [Dropbox](https://www.dropbox.com/scl/fo/9l80t45qtxs66rrllcz9r/ALX4XvCn98vMnWUGiL1W_CY?rlkey=x5ii5yj1apzkzwg371pu6erjd&st=7f09lp98&dl=0)
- Place them in the root of your project directory.
- Combine them using:

        python combine_news.py

### 3. Train and Evaluate the Model 
      python fake_news_detector.py

### 4. Use the trained model to predict new articles
      python predict_news.py
  
## Screenshots

Class Distribution

<img width="600" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/593be401-0dc3-4d79-afeb-f3aa33c6f913" />

Confusion Matrix 

<img width="600" height="400" alt="confusion matrix" src="https://github.com/user-attachments/assets/0ae77583-c4d6-412a-81b9-0f2994410421" />

Confusion Matrix with ROC Curve

<img width="600" height="400" alt="roc curve" src="https://github.com/user-attachments/assets/ead3f57e-3dd4-4932-a78a-3b179b235c72" />

