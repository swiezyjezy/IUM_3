from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from dotenv import load_dotenv
import os

from data_prep import load_data, split_data
import joblib

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
MODEL_TYPE = os.getenv("MODEL_TYPE")

df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_SEED)

if MODEL_TYPE == "logistic_regression":
    model = LogisticRegression()
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

pipeline_LR = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
])

pipeline_LR.fit(X_train, y_train)
joblib.dump(pipeline_LR, "models/logistic_regression_model.pkl")
