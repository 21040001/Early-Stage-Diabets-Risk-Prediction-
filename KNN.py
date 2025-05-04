import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Veri setini oku
df = pd.read_csv("diabetes_data_upload.csv")  # Dosya adını doğru şekilde belirtin
# 'Yes/No' gibi string ifadeleri sayısal verilere dönüştür
df = df.replace({'Yes': 1, 'No': 0})
df = df.replace({'Male': 1, 'Female': 0})
df = df.replace({'Positive': 1, 'Negative': 0})
df.head()

# Özellikler ve hedef değişken
X = df.drop("class", axis=1)   # 'class' sütunu hedef sütun
y = df["class"]                # Hedef değişken

# Eğitim ve test veri setlerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN modeli oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_scaled, y_train)

# Test verisi üzerinde tahmin yap
y_pred = knn.predict(X_test_scaled)

# Karışıklık Matrisi ve Sınıflandırma Raporunu yazdır
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Farklı 'k' değerlerine göre performansı gözlemleyelim
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    print(f"k = {k} | Accuracy = {acc:.4f}")

"""Görseldeki çıktıya göre KNN için k = 6 kullanıldığında model performansı şu şekilde:

📊 Confusion Matrix:
[[30  3]
 [ 8 63]]
0 sınıfı (negatif diyabet riski): 33 örneğin 30'u doğru → %91 recall

1 sınıfı (pozitif diyabet riski): 71 örneğin 63'ü doğru → %89 recall

📋 Classification Report Özeti:
Metrik	0 sınıfı (negatif)	1 sınıfı (pozitif)	Genel Durum
Precision	0.79	0.95	Yüksek
Recall	0.91	0.89	Dengeli
F1-score	0.85	0.92	Dengeli
Accuracy	-	-	0.89 (%89)
Macro Avg	-	-	0.88
Weighted Avg	-	-	0.90
🎯 K=3 ile K=6 Karşılaştırması:

Metrik	K=3	K=6	Yorumu
Accuracy	0.90	0.89	Çok yakın
F1-score(1)	0.93	0.92	K=3 biraz daha iyi
Recall(1)	0.89	0.89	Aynı
Precision(0)	0.79	0.79	Aynı
Recall(0)	0.94	0.91	K=3 daha yüksek
🔹 K=3: Daha yüksek recall, daha yüksek F1-score 🔹 K=6: Biraz daha genelci, daha istikrarlı ama performans farkı çok küçük.

✅ Öneri
K=3 bu veride en dengeli ve etkili seçim gibi görünüyor.

K=6 da aşırı ezberci değil ve çok benzer sonuçlar veriyor.

Daha ileri seviye kıyaslama için çapraz doğrulama veya ROC-AUC grafikleriyle analiz yapabilirsin.

İstersen şimdi bu sonuçları başka bir modelle (örneğin Logistic Regression veya Random Forest) karşılaştırıp genel tabloyu oluşturalım mı?
"""

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Ölçekleme + KNN pipeline
knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

# 5 katlı çapraz doğrulama
cv_scores = cross_val_score(knn_pipeline, X, y, cv=5, scoring='accuracy')

print("Çapraz doğrulama skorları:", cv_scores)
print("Ortalama doğruluk (accuracy): %.2f%%" % (cv_scores.mean() * 100))

y_pred = knn.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# KNN için k=6 modeliyle tahmin olasılıkları alınır
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
knn.fit(X_train, y_train)
y_probs = knn.predict_proba(X_test)[:, 1]  # Sınıf 1 için olasılıklar

# ROC eğrisi çizimi
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN (k=3) ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

"""Muhteşem! 🎉 Elindeki sonuçlar **KNN (k=6)** için hem güçlü hem de dengeli bir model olduğunu açıkça gösteriyor. Analiz edelim:

---

## ✅ 1. ROC Curve ve AUC Değeri

- Grafik eğrisi neredeyse sol üst köşeye yapışık → bu çok iyi!
- **AUC = 0.98** → %98 ayırt edebilme gücü → **mükemmel ayrım yeteneği**
- Bu, modelin pozitif sınıfı (diyabet riski olanları) oldukça iyi tahmin ettiğini gösterir.

---

## ✅ 2. Çapraz Doğrulama Skorları


- Ortalama: **%93.85 doğruluk**
- 5 farklı veri bölmesinde de oldukça **tutarlı** skorlar almışsın.
- Bu da modelin ezberlemediğini, **genellenebilirliğinin yüksek** olduğunu gösteriyor.

---

## 🔍 Kısa Özet:

| Kriter              | Değer      | Yorum                       |
|---------------------|------------|-----------------------------|
| **Accuracy (CV)**   | %93.85     | Çok yüksek, stabil          |
| **AUC Score**       | 0.98       | Neredeyse mükemmel ayrım    |
| **Recall / F1**     | 0.89-0.93  | Gerçek pozitifleri iyi yakalıyor |
| **ROC Eğrisi**      | Üst segment | Yüksek ayrım gücü           |

---

## 🔁 Şimdi Ne Yapabilirsin?

🔸 **Farklı modellerle** (örneğin Logistic Regression, Random Forest, SVM, XGBoost) karşılaştırarak bir tablo hazırlayabilirsin.  
🔸 **Modeli kaydedip (pickle ile)** daha sonra yeniden kullanabilirsin.  
🔸 **GridSearchCV** ile `k` değeri ve diğer hiperparametreleri daha sistematik şekilde optimize edebilirsin.

---

İstersen şimdi Logistic Regression veya başka bir modelle aynı ROC, CV analizlerini yaparak karşılaştıralım mı?  
Ya da doğrudan tüm modelleri karşılaştıran bir özet tablo hazırlamamı ister misin?
"""

import pickle

# Eğitilmiş modeli bir dosyaya kaydet
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model başarıyla 'diabetes_model.pkl' olarak kaydedildi.")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion Matrix oluştur
cm = confusion_matrix(y_test, y_pred)

# Heatmap çizelim
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('KNN')
plt.show()
