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
df = pd.read_csv("diabetes_data_upload.csv")  # Dosya adı farklıysa, dosyanızın adını buraya girin.
df.head()

# Veri setinin genel bilgilerini inceleyelim
df.info()

# Eksik verilerin sayısını görelim
df.isnull().sum()

# 'Yes/No' gibi string ifadeleri sayısal verilere çeviriyoruz
df = df.replace({'Yes': 1, 'No': 0})
df = df.replace({'Male': 1, 'Female': 0})
df = df.replace({'Positive': 1, 'Negative': 0})
df.head()

# Hedef sütun: class (Diyabet var mı yok mu?)
X = df.drop("class", axis=1)  # 'class' sütununu çıkarıyoruz, geri kalan tüm sütunlar X olacak
y = df["class"]               # 'class' sütunu hedef değişken

# İlk 5 satırı kontrol edelim
X.head()

# Veriyi %80 eğitim, %20 test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test veri boyutlarını görelim
print("Eğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)

# Lojistik regresyon modelini oluştur
model = LogisticRegression(max_iter=100)

# Modeli eğit
model.fit(X_train, y_train)

# Test verisi ile tahmin yapalım
y_pred = model.predict(X_test)

# İlk 10 tahmine bakalım
print("Tahminler:", y_pred[:10])
print("Gerçek değerler:", y_test.values[:10])

# Doğruluk oranını hesaplayalım
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk (Accuracy):", accuracy)

# Karışıklık matrisi
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))

# Ayrıntılı sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Özelliklerin önemini görelim
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print(coeff_df)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Özellikler ve katsayılar (manuel girildi)
data = {
    "Feature": [
        "Polydipsia", "Polyuria", "Irritability", "Genital thrush", "partial paresis", "sudden weight loss",
        "visual blurring", "weakness", "Polyphagia", "Age", "Alopecia", "muscle stiffness", "Obesity",
        "delayed healing", "Itching", "Gender"
    ],
    "Coefficient": [
        2.634848, 2.578316, 1.470152, 1.336485, 1.014970, 0.903378,
        0.774093, 0.531706, 0.455958, -0.036840, -0.158542, -0.176877, -0.186113,
        -0.563419, -1.203638, -2.385263
    ]
}

# DataFrame oluştur ve sıralama yap
df_coef = pd.DataFrame(data)
df_coef_sorted = df_coef.sort_values(by="Coefficient", ascending=False)

# Grafik çizimi
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=df_coef_sorted, palette="coolwarm")
plt.title("Lojistik Regresyon - Özelliklerin Katsayı Etkisi")
plt.xlabel("Katsayı Değeri (Etki Gücü)")
plt.ylabel("Özellik (Feature)")
plt.axvline(0, color="black", linestyle="--")
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

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
plt.title('Logistic Regresyon')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Modelin tahminlerini alıyoruz
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Eğer model predict_proba kullanıyorsa

# FPR ve TPR'yi hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# AUC'yi hesapla
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
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

"""# Modelimizde Overfitting olup olmadığını kontrol edelim"""

# Eğer Eğitim > Test farkı çok fazlaysa → Overfitting olabilir.
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Eğitim Doğruluğu:", train_acc)
print("Test Doğruluğu:", test_acc)
# Değerler yaklaşık aynı olduğu için Overfitting yok

"""🔍 **Mükemmel!** Hem eğitim doğruluğun **%92.3**, hem test doğruluğun **%92.3** çıkmış. Bu şu anlama gelir:

---

##  Sonuç: Overfitting Yok

  Eğitim ve test doğruluğu neredeyse eşit** | Bu, modelin sadece eğitim verisini ezberlemediğini; yeni, görmediği verilerde de benzer başarı gösterdiğini kanıtlar. |
|  **AUC = 0.97** | Model güçlü, duyarlı ve genelleyebilir durumda. |
|  **Yorum:** | Model **overfitting yapmıyor**, aynı zamanda **underfitting de değil** – tam kararında bir öğrenme düzeyi var. |

---

##  Şu An Modelin Durumu:
- **Genelleme gücü yüksek**
- **Ezberlememiş**
- **Yeni verilere karşı sağlam**
- Gerçek dünya uygulamaları için oldukça uygun başlangıçta

---

İstersen şimdi:
- Diğer algoritmalarla karşılaştıralım mı?
- Yoksa model raporu, PDF veya sunum dosyası oluşturalım mı?

Nasıl ilerleyelim?

# Cross-Validation (Çapraz Doğrulama)
Farklı veri dilimlerinde model kararlı çalışıyor mu görmek için yapılır. 
"""

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Modeli tanımla
model_cv = LogisticRegression(max_iter=1000)

# 5-Fold Cross-Validation ile doğruluk hesapla
scores = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy')

# Sonuçları yazdır
print("Her fold için doğruluk skorları:", scores)
print("Ortalama doğruluk:", scores.mean())
print("Standart sapma:", scores.std())

""" **Mükemmel bir sonuç!** Cross-validation çıktını değerlendirelim:
##  Cross-Validation Sonuçları

| Fold (kat) | Doğruluk |
|------------|----------|
| Fold 1     | 0.8942   |
| Fold 2     | 0.9327   |
| Fold 3     | 0.8750   |
| Fold 4     | 0.9712   |
| Fold 5     | 0.9519   |

---

###  **Ortalama Doğruluk: `0.925` (%92.5)**
- Bu, daha önce hesapladığımız test doğruluğuyla (%92.3) neredeyse **bire bir uyuşuyor**.
- Yani model **veri seti içinde tutarlı çalışıyor** ve **şansa bağlı sapma göstermiyor**
