import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

def load_data(csv_path='news.csv'):
    df = pd.read_csv(csv_path)
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    return df

def plot_class_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution: Fake (0) vs Real (1)')
    plt.xticks([0, 1], ['Fake', 'Real'])
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_and_evaluate_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)
    y_scores = model.predict_proba(x_test_tfidf)[:,1]  # For ROC curve

    print("Classification Report:\n", classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)

    # Save the trained model and vectorizer
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved!")

def main():
    data = load_data()
    plot_class_distribution(data)
    train_and_evaluate_model(data)

if __name__ == '__main__':
    main()
