import joblib
from dotenv import load_dotenv
import os

from sklearn.metrics import accuracy_score, classification_report
from data_prep import load_data, split_data

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))

df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_SEED)

model = joblib.load("models/logistic_regression_model.pkl")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report)
