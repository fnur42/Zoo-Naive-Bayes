import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Veriyi yükleme
veri = pd.read_csv("C:/Users/User/Downloads/hayvanatbahcesi.csv", encoding='unicode_escape')

# Giriş ve çıkış değişkenlerini belirleme
X = veri.drop(["sinifi"], axis=1)
y = veri["sinifi"]

# Kategorik sütunları belirleme
categorical_columns = X.select_dtypes(include=['object']).columns

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

# Pipeline oluşturma
categorical_pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))  # seyrek olmayan bir matris elde etmek için sparse=False
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipe, categorical_columns)
    ])

# Bayes öğrenme modeli için boru hattı oluşturma
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', CategoricalNB())
])

# Modeli eğitme
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Doğruluk hesaplama
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
