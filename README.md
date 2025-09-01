Makine Öğrenmesi ile Erken Evre Diyabet Riskinin Sınıflandırılması
Bu proje, erken evre diyabet riskini sınıflandırmak için çeşitli makine öğrenmesi algoritmalarını değerlendirmektedir. Çalışma, 520 hastadan toplanan 16 farklı semptom özelliğini kullanarak yedi farklı algoritmanın performansını karşılaştırmaktadır.

📊 Veri Seti
Veri seti, Hindistan'dan toplanan ve 16'sı bağımsız değişken, 1'i hedef değişken (diyabet durumu: Positive/Negative) olmak üzere toplam 17 özellik içermektedir. Özellikler arasında yaş, cinsiyet, polidipsi (aşırı susama), yorgunluk, bulanık görme, obezite gibi semptomlar bulunmaktadır.

Veri ön işleme aşamasında kategorik değişkenler ikili forma dönüştürülmüş (Yes:1, No:0; Male:1, Female:0) ve veri seti %80 eğitim, %20 test olarak ayrılmıştır.

🧠 Kullanılan Algoritmalar
Aşağıdaki yedi makine öğrenmesi algoritması uygulanmış ve performansları değerlendirilmiştir:

Naive Bayes

Lojistik Regresyon

Karar Ağacı

Rastgele Orman

K-En Yakın Komşu (K-NN)

Doğrusal Regresyon

Yapay Sinir Ağı (YSA)

📈 Performans Metrikleri
Modeller aşağıdaki metrikler kullanılarak değerlendirilmiştir:

Doğruluk (Accuracy)

Kesinlik (Precision)

Hassasiyet (Recall/Sensitivity)

F1-Skoru

AUC (ROC Eğrisi Altında Kalan Alan)

🏆 Sonuçlar
Model	Doğruluk	Kesinlik	Hassasiyet	AUC	F1-Skoru
Naive Bayes	0.91	0.90	0.90	0.95	0.90
K-NN	0.93	0.87	0.90	0.98	0.88
Lojistik Regresyon	0.92	0.92	0.90	0.97	0.91
Karar Ağacı	0.98	0.97	0.99	0.98	0.98
Rastgele Orman	0.99	0.99	0.99	1.00	0.99
Doğrusal Regresyon	0.60	0.92	0.90	0.98	0.91
Yapay Sinir Ağı	0.97	0.96	0.98	1.00	0.97
En yüksek performans Rastgele Orman algoritması tarafından elde edilmiştir.

📌 Çıkarımlar
Rastgele Orman ve Yapay Sinir Ağı modelleri en yüksek doğruluk ve AUC değerlerine ulaşmıştır.

Doğrusal Regresyon, ikili sınıflandırma için uygun olmadığından en düşük performansı göstermiştir.

Semptom tabanlı verilerle erken diyabet teşhisi yüksek doğrulukla mümkündür.

🔮 Öneriler
Derin öğrenme ve XGBoost gibi daha gelişmiş algoritmaların denenmesi

Daha genç ve cinsiyet açısından dengeli veri setleriyle çalışılması

Gerçek zamanlı tarama sistemlerine entegrasyon için mobil uyumlu modeller geliştirilmesi

📄 Etik Beyan
Bu çalışmada kimliksizleştirilmiş ve halka açık bir veri seti kullanılmıştır. Kişisel veri içermediğinden etik kurul onayı gerekmemektedir.

👥 Yazarlar
Kağan Güner - Gazi Üniversitesi

Davronbek Abdurazzokov - Gazi Üniversitesi

📚 Kaynaklar
Makale içerisinde atıf yapılan kaynaklar listesi raporun son sayfasında mevcuttur.
