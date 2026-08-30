import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

def train():
    df = pd.read_csv("../resources/jigsaw_snapshot/data.csv")
    df["toxic"] = (df["target"] >= 0.55)
    y = df["toxic"]
    X = df["comment_text"]

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', svm.LinearSVC(random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred_val = pipeline.predict(X_val)

    print("Classification Report:")
    print(classification_report(y_val, y_pred_val))

    opcion = input("\n¿Deseas evaluar en Test y guardar el modelo? (s/n): ").strip().lower()

    if opcion == "s":
        print("\nEvaluando en conjunto de Test...")
        y_pred_test = pipeline.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

        package = {
            "pipeline": pipeline,
            "f1_macro": round(f1_score(y_test, y_pred_test, average='macro'), 4)
        }
        output_path = "toxic_linear_svc.joblib"
        joblib.dump(package, output_path)
        print(f"\nModelo guardado exitosamente en '{output_path}'")
    else:
        print("\nOperación cancelada. El modelo no fue guardado.")

if __name__ == '__main__':
    train()