import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Veri setini yükleyelim
df = pd.read_csv("diabetes_data_upload.csv")  # Dosya adını doğru yazdığınızdan emin olun
df.head()

# 'Yes/No' gibi string ifadeleri sayısal verilere dönüştür
df = df.replace({'Yes': 1, 'No': 0})
df = df.replace({'Male': 1, 'Female': 0})
df = df.replace({'Positive': 1, 'Negative': 0})
df.head()

# Özellikler (X) ve hedef değişken (y)
X = df.drop("class", axis=1)  # class = hedef sütun
y = df["class"]               # hedef değişken

# Eğitim ve test veri setlerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirelim (StandardScaler ile)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modelini oluşturalım
model = GaussianNB()

# Modeli eğitelim
model.fit(X_train_scaled, y_train)

# Test verisiyle tahmin yapalım
y_pred = model.predict(X_test_scaled)

# Başarı metriklerini yazdıralım
print("✅ Doğruluk Oranı (Accuracy):", accuracy_score(y_test, y_pred))
print("\n📄 Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\n🧮 Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# Modeli kaydedelim (pickle kullanarak)
with open("diabetes_modelNaivesBayes.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model başarıyla 'diabetes_modelNaivesBayes.pkl' olarak kaydedildi.")

# Confusion Matrix görselleştirmesini yapalım
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Naive Bayes Karışıklık Matrisi')
plt.show()

# Gereksiz sütunları çıkararak modeli iyileştirelim
X_filtered = df.drop(["Age", "Obesity", "class"], axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Veriyi tekrar ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Yeni Naive Bayes modelini tekrar eğitelim
model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Test verisi ile tahmin yapalım
y_pred = model.predict(X_test_scaled)

# Başarı metriklerini tekrar yazdıralım
print("✅ Doğruluk Oranı (Accuracy):", accuracy_score(y_test, y_pred))
print("\n📄 Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\n🧮 Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# Modelin başarımını daha iyi görmek için ROC eğrisini çizelim
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # Modelin olasılık tahminlerini alıyoruz
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# AUC'yi hesaplayalım
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizelim
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random model için diagonal çizgi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
