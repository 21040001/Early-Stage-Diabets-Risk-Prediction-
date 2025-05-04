import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import pickle

# CSV dosyasını yükle
df = pd.read_csv("diabetes_data_upload.csv")  # Dosya adını doğru şekilde belirtin
df.head()

# 'Yes/No' gibi string ifadeleri sayısal verilere dönüştür
df = df.replace({'Yes': 1, 'No': 0})
df = df.replace({'Male': 1, 'Female': 0})
df = df.replace({'Positive': 1, 'Negative': 0})
df.head()

# Özellikler (X) ve hedef değişken (y)
X = df.drop("class", axis=1)   # 'class' sütunu hedef sütun
y = df["class"]                # Hedef değişken

# Eğitim ve test veri setlerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli oluştur (Random Forest)
model = RandomForestClassifier(random_state=16)

# Modeli eğit
model.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = model.predict(X_test)

# Sonuçları değerlendir
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Modelin performansını daha ayrıntılı görmek için sınıf dengesini kontrol edelim
print("Eğitim seti sınıf dağılımı:")
print(y_train.value_counts())
print("\nTest seti sınıf dağılımı:")
print(y_test.value_counts())

# Confusion Matrix çizelim
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Random Forest Karışıklık Matrisi')
plt.show()

# Cross-validation ile modelin tutarlılığını test edelim
scores = cross_val_score(model, X_train, y_train, cv=5)
print("5-Fold Cross-Validation Skorları:", scores)
print("Ortalama CV doğruluğu:", scores.mean())

# ROC Eğrisini çizelim
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Sınıf 1 için olasılıkları alalım
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

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

# Modeli kaydetmek için pickle kullanıyoruz
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model başarıyla kaydedildi: diabetes_model.pkl")
