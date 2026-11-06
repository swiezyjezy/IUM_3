import time

from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
import numpy as np
import torch

# ------------- ZAD 1 ------------

# ładowanie danych
wine_data = load_wine(as_frame=True)
df = wine_data.frame

# podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=32)

# ------------- ZAD 2 ------------

pipeline_LR = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# trenowanie modelu LR
time_start = time.time()
pipeline_LR.fit(X_train, y_train)
time_end = time.time()
time_elapsed = time_end - time_start

# ewaluacja modelu LR
y_pred_lr = pipeline_LR.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_lr)
print(f"--- Logistic Regression ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Time elapsed: {time_elapsed:.4f} seconds")

# ------------- ZAD 3 ------------

# skalowanie danych dla PyTorch
scaler_pt = StandardScaler()
X_train_scaled = scaler_pt.fit_transform(X_train)
X_test_scaled = scaler_pt.transform(X_test)

# konwersja na tensory
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = len(np.unique(df['target']))

# definicja sieci neuronowej w PyTorch

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, output_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def fit(self, X, y, epochs=60):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

# tworzenie instancji i trenowanie sieci neuronowej
torch.manual_seed(32)
net = Network()
start_time = time.time()
net.fit(X_train_t, y_train_t, epochs=80)
end_time = time.time()
training_time = end_time - start_time

# ewaluacja PyTorch
net.eval()
with torch.no_grad():
    outputs_test = net(X_test_t)
    _, predicted = torch.max(outputs_test, 1)
    accuracy_pt = accuracy_score(y_test_t.numpy(), predicted.numpy())


print("\n--- PyTorch ---")
print(f"Accuracy: {accuracy_pt:.2f}")
print(f"Time elapsed: {training_time:.4f} seconds")

# ------------- ZAD 4 ------------   

# podzial danych na 5 foldów
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# cross-validation dla modelu LR
scores = cross_val_score(pipeline_LR, X_train, y_train, cv=kf, scoring="accuracy")

print("\n--- Cross-validation Logistic Regression ---")
for fold, score in enumerate(scores, 1):
    print(f"Fold {fold}: {score:.2f}")
print(f"Mean: {scores.mean():.2f}, Std: {scores.std():.2f}")

# ------------- ZAD 5 ------------

cv_scores_pt = []

X_all = df.drop(columns=['target'])
y_all = df['target']

for train_idx, test_idx in kf.split(X_all):

    # podzial danych na foldy według indeksów z KFold
    X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]
    
    # skalowanie dla każdego foldu
    fold_scaler = StandardScaler()
    X_tr_s = fold_scaler.fit_transform(X_tr)
    X_te_s = fold_scaler.transform(X_te)

    # konwersja na tensory
    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.long)
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32)
    y_te_t = torch.tensor(y_te.values, dtype=torch.long)
    
    # nowa instancja sieci dla każdego foldu
    torch.manual_seed(32)
    fold_net = Network()
    fold_net.fit(X_tr_t, y_tr_t, epochs=80)
    
    # ewaluacja
    fold_net.eval()
    with torch.no_grad():
        test_outputs = fold_net(X_te_t)
        _, preds = torch.max(test_outputs, 1)
        acc = accuracy_score(y_te_t.numpy(), preds.numpy())
        cv_scores_pt.append(acc)

cv_scores_pt = np.array(cv_scores_pt)

print(f"\n--- Cross-validation PyTorch ---")
for fold, score in enumerate(cv_scores_pt, 1):
    print(f"Fold {fold}: {score:.2f}")
print(f"Mean: {cv_scores_pt.mean():.2f} Std: {cv_scores_pt.std():.2f}")
