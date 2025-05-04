# Veri işleme ve görselleştirme kütüphaneleri
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Makine öğrenmesi modülleri
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# CSV dosyasını oku
data = pd.read_csv("./diabetes_data_upload.csv")
data.head()

# Veri setinin genel bilgilerine bakalım
data.info()

# Eksik veri sayısını kontrol edelim
data.isnull().sum()

# Kategorik verileri sayısal verilere dönüştürüyoruz
data = data.replace({'Yes': 1, 'No': 0})
data = data.replace({'Male': 1, 'Female': 0})
data = data.replace({'Positive': 1, 'Negative': 0})
data.head()

# 'class' sütununu hedef değişken olarak belirliyoruz
X = data.drop("class", axis=1)  # Özellikler (semptomlar vs.)
y = data["class"]               # Hedef değişken (diyabet durumu)

# İlk 5 satırı inceleyelim
X.head()

# Veriyi %80 eğitim, %20 test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test veri boyutlarını kontrol edelim
print("Eğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)

# Karar ağacı modelini oluşturalım ve eğitelim
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42)
model.fit(X_train, y_train)

# Test verisiyle tahmin yapalım
y_pred = model.predict(X_test)

# Sonuçları yazdıralım
print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# Özelliklerin önem düzeyini görselleştirelim
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel("Önem Düzeyi")
plt.ylabel("Özellikler")
plt.title("Özelliklerin Önem Grafiği")
plt.show()

# Veriden bazı gereksiz sütunları çıkaralım
X = data.drop(['weakness', 'sudden weight loss', 'Polyphagia', 'visual blurring', 'partial paresis', 'class'], axis=1)
y = data['class']

# Eğitim ve test setlerini tekrar ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini yeniden oluşturalım ve eğitelim
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sonuçları yazdıralım
print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# Özelliklerin önem düzeyini görselleştirelim
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel("Önem Düzeyi")
plt.ylabel("Özellikler")
plt.title("Özelliklerin Önem Grafiği")
plt.show()

# 'Itching' sütununu çıkaralım ve modelimizi tekrar eğitelim
X = data.drop(['Itching', 'class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini tekrar oluşturalım ve eğitelim
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# 'Weakness' ve 'Polyphagia' sütunlarını çıkaralım ve modelimizi tekrar eğitelim
X = data.drop(['weakness', 'Polyphagia', 'partial paresis', 'class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini tekrar oluşturalım ve eğitelim
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# 'sudden weight loss' sütununu çıkaralım ve modelimizi tekrar eğitelim
X = data.drop(['sudden weight loss', 'class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini tekrar oluşturalım ve eğitelim
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# Modelin karar ağacını görselleştirelim
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Ağacı export edelim
dot_data = export_graphviz(
    model,
    feature_names=X.columns,
    class_names=["Negative", "Positive"],
    filled=True, rounded=True,
    special_characters=True
)

# Görselleştirelim
graph = graphviz.Source(dot_data)
graph.render("diabetes_decision_tree", format="png", cleanup=False)  # PNG olarak kaydeder
graph  # Jupyter'de görsel olarak gösterir

# Ağaç yapısını çizdirelim (max_depth=3)
dot_data = export_graphviz(
    model,
    feature_names=X.columns,
    class_names=["Negative", "Positive"],
    filled=True, rounded=True,
    special_characters=True,
    max_depth=3
)

# Görselleştir
graph = graphviz.Source(dot_data)
graph.render("diabetes_decision_tree", format="png", cleanup=False)  # PNG olarak kaydeder
graph  # Jupyter'de görsel olarak gösterir

# Karar ağacının daha sade haliyle görselleştirilmesi
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["Negative", "Positive"], filled=True)
plt.show()
