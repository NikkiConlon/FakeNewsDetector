# Fake News Detector

This project uses a logistic regression model to classify news articles as fake or real based on their text content. It covers data preprocessing, model training, evaluation, and prediction.

## Features

- Uses TF-IDF vectorization for text feature extraction  
- Logistic Regression for classification  
- Model evaluation with precision, recall, and F1-score  
- Confusion matrix and classification reports for performance insights  
- Easily extendable to other datasets or models  
- Predict fake news from new text inputs  

## Notes
- Due to GitHub size limits, datasets are not included. Download them here: [Dropbox](https://www.dropbox.com/scl/fo/9l80t45qtxs66rrllcz9r/ALX4XvCn98vMnWUGiL1W_CY?rlkey=x5ii5yj1apzkzwg371pu6erjd&st=7f09lp98&dl=0)
- Place all datasets in the project root before running scripts.

## How to Run

1. Download the datasets (`Fake.csv` and `True.csv`) and place them in the project root.  
2. Run `combine_news.py` to merge datasets into `news.csv`.  
3. Train and evaluate the model using:  
   python fake_news_detector.py
4. Use the trained model to predict new articles:
   python predict_news.py
5. Replace the sample text in predict_news.py with your own news article to test.

## Screenshots

Class Distribution

<img width="600" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/593be401-0dc3-4d79-afeb-f3aa33c6f913" />

Confusion Matrix 

<img width="600" height="400" alt="confusion matrix" src="https://github.com/user-attachments/assets/0ae77583-c4d6-412a-81b9-0f2994410421" />

Confusion Matrix with ROC Curve

<img width="600" height="400" alt="roc curve" src="https://github.com/user-attachments/assets/ead3f57e-3dd4-4932-a78a-3b179b235c72" />
