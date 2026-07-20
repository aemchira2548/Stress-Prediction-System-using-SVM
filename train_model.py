from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student-lifestyle-and-stress-dataset.csv"
MODEL_FILE = BASE_DIR / "svm_stress_model.pkl"
GRAPH_FILE = BASE_DIR / "confusion_matrix.png"


def main():
    if not DATA_FILE.exists():
        print("ไม่พบไฟล์ CSV")
        return

    df = pd.read_csv(DATA_FILE)

    print("ขนาดข้อมูล:", df.shape)
    print("\nค่าว่างแต่ละคอลัมน์")
    print(df.isnull().sum())

    df = df.drop_duplicates()
    df = df.dropna(subset=["Stress_Level"])

    X = df.drop(columns=["Stress_Level"])
    y = df["Stress_Level"].astype(int)

    categorical_features = ["Student_Type"]

    numeric_features = [
        "Sleep_Hours",
        "Study_Hours",
        "Social_Media_Hours",
        "Attendance",
        "Exam_Pressure",
        "Family_Support",
        "Month",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nกำลังฝึกโมเดล...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Stress", "Stress"],
    )

    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300)
    plt.show()

    joblib.dump(model, MODEL_FILE)

    print("\nบันทึกโมเดลแล้ว:", MODEL_FILE)
    print("บันทึกกราฟแล้ว:", GRAPH_FILE)


if __name__ == "__main__":
    main()