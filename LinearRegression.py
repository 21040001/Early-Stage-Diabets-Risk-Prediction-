import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Veri setini yükle
df = pd.read_csv("diabetes_data_upload.csv")  # Dosya adı farklıysa, kendi dosya adınızı kullanabilirsiniz.

# Yes/No gibi string ifadeleri sayısal verilere çeviriyoruz
df = df.replace({'Yes': 1, 'No': 0})
df = df.replace({'Male': 1, 'Female': 0})
df = df.replace({'Positive': 1, 'Negative': 0})
df.head()

# Gereksiz sütunu kaldırıyoruz
df = df.drop(['sudden weight loss'], axis=1)

# Özellikler (X) ve hedef (y) değişkenlerini ayırıyoruz
X = df.drop("class", axis=1)   # 'class' hedef değişkeni
y = df["class"]                # Hedef değişken

# Eğitim ve test verisini ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

# Lineer Regresyon modelini oluşturuyoruz
model = LinearRegression()

# Modeli eğitim verisiyle eğitiyoruz
model.fit(X_train, y_train)

# Test verisiyle tahmin yapıyoruz
y_pred = model.predict(X_test)

# Olasılıkları 0.5 eşik değeri ile sınıflara dönüştürüyoruz
y_pred_class = (y_pred > 0.5).astype(int)

# Karışıklık Matrisi'ni oluşturuyoruz
cm = confusion_matrix(y_test, y_pred_class)

from sklearn.metrics import mean_squared_error, r2_score

# Ortalama kare hatası (MSE) hesaplıyoruz
mse = mean_squared_error(y_test, y_pred)

# R-kare skoru (Modelin başarı oranı)
r2 = r2_score(y_test, y_pred)

# Sonuçları yazdırıyoruz
print("Ortalama Kare Hatası (MSE):", mse)
print("R-Kare Skoru:", r2)

"""# Lineer regresyon modelini denedik ve başarı oranı %58 ile düşük çıktı.
# Bu başarısız sonuç nedeniyle, gereksiz sütunları çıkararak başarıyı artırmayı hedefliyoruz."""

# 'sudden weight loss' sütununu çıkardık ve başarı oranında %4'lük bir artış gördük.
# Ancak bu oran hala tatmin edici değil, bu yüzden farklı algoritmalar denemeye karar verdik.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Lojistik Regresyon modelini oluşturuyoruz
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Test verisiyle tahmin yapıyoruz
y_pred = model.predict(X_test)

# Başarı metriklerini yazdırıyoruz
print(" Doğruluk Oranı (Accuracy):", accuracy_score(y_test, y_pred))
print("\n Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\n Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

# Başarı skoru öncesi ve sonrası karşılaştırmasını görselleştiriyoruz
r2_before = 0.58  # Sütun çıkarmadan önceki başarı
r2_after = 0.65   # Sütun çıkardıktan sonraki başarı

plt.figure(figsize=(6, 4))
plt.bar(['Önce', 'Sonra'], [r2_before, r2_after], color=['red', 'green'])
plt.ylim(0, 1)
plt.ylabel("R² Başarı Skoru")
plt.title("Gereksiz Sütun Çıkarmanın Model Başarısına Etkisi")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ROC eğrisini çizmek için gerekli işlemler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Modelin tahminlerini alıyoruz
y_pred = model.predict(X_test)

# Regresyon çıktısını 0 ve 1 sınıflarına dönüştürüyoruz
y_pred_class = (y_pred > 0.5).astype(int)

# FPR ve TPR'yi hesaplıyoruz
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# AUC'yi hesaplıyoruz
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiziyoruz
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

# Modelin sınıf tahminlerini almak için predict() fonksiyonunu kullanalım
y_pred = model.predict(X_test)

# Başarı metriklerini tekrar hesaplıyoruz
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Sonuçları yazdıralım
accuracy, classification_rep, conf_matrix
