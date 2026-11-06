from src.data_prep import load_data, split_data
import pandas as pd
import pytest

def test_load_data():
    # test sprawdza czy funkcja load_data poprawnie wczytuje dane z pliku CSV
    df = load_data("data/wine.csv")
    assert not df.empty
    assert "target" in df.columns

def test_split_data_proportions():
    # test sprawdza czy split_data zwraca zbiory o 
    # oczekiwanym rozmiarze (proporcja test_size)

    df = load_data("data/wine.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=32)
    
    total_len = len(df)
    assert len(X_test) == pytest.approx(total_len * 0.2, abs=2)
    assert len(X_train) == pytest.approx(total_len * 0.8, abs=2)

def test_split_data_sizes():
    # test sprawdza czy suma rozmiarów zbioru treningowego 
    # i testowego równa się rozmiarowi całego zbioru

    df = load_data("data/wine.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=32)
    
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

def test_split_data_invalid_input():
    # Test sprawdza zachowanie przy niepoprawnym wejściu: brak kolumny 'target'

    df_invalid = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5], 
        "feature2": [6, 7, 8, 9, 10]
    })
    with pytest.raises(KeyError):
        split_data(df_invalid, test_size=0.2, random_state=42)