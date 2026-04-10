# Meyve Sınıflandırma (Transfer Learning & Fine-Tuning)

**Bu proje, derin öğrenme tekniklerini öğrenmek ve farklı CNN modellerinin performansını karşılaştırmak amacıyla geliştirilmiş bir çalışmadır.**

Bu projede, farklı derin öğrenme modelleri kullanılarak meyve görsellerinin sınıflandırılması gerçekleştirilmiştir.  
Amaç, **birden fazla CNN mimarisinin performansını karşılaştırmak** ve en iyi sonucu veren modeli belirlemektir.

---

##  Kullanılan Yöntemler

- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Çok sınıflı sınıflandırma (Multi-class classification)

---
## Kullanılan Kütüphaneler

Bu projede aşağıdaki Python kütüphaneleri kullanılmıştır:

### 🔹 Derin Öğrenme
- tensorflow
- keras (TensorFlow backend ile)

### 🔹 Veri İşleme
- numpy
- pandas

### 🔹 Görselleştirme
- matplotlib
- seaborn

### 🔹 Model Değerlendirme
- scikit-learn
  - confusion_matrix
  - accuracy_score
  - precision_score
  - recall_score
  - f1_score
  - classification_report
  - roc_auc_score
  - roc_curve
  - auc
  - label_binarize
## Kullanılan Modeller

Projede aşağıdaki hazır (pre-trained) modeller kullanılmıştır:

- MobileNetV2
- EfficientNetB0
- ResNet50
- VGG16
- InceptionV3

---

##  Veri Seti

- Meyve görsellerinden oluşan özel bir veri seti kullanılmıştır.
## ⚙️ Veri Ön İşleme ve Veri Üretimi (Data Generator)

### 🔹 Data Augmentation (Veri Artırma)

Eğitim veri setinin çeşitliliğini artırmak ve modelin genelleme yeteneğini geliştirmek amacıyla veri artırma teknikleri uygulanmıştır.

Uygulanan işlemler:
- Normalizasyon (`rescale=1./255`)
- Rastgele döndürme (`rotation_range=30`)
- Yakınlaştırma (`zoom_range=0.2`)
- Yatay çevirme (`horizontal_flip=True`)

Test veri setine herhangi bir veri artırma uygulanmamıştır.  
Bu sayede model, gerçek ve değiştirilmemiş veriler üzerinde değerlendirilmiştir.



### 🔹 Data Generator Yapısı

Keras `ImageDataGenerator` yapısı kullanılarak görseller klasörlerden okunmuş ve model eğitimine uygun hale getirilmiştir.

- Görseller `224x224` boyutuna yeniden ölçeklendirilmiştir
- Veriler batch (grup) halinde modele verilmiştir (`batch_size=32`)
- Çok sınıflı sınıflandırma için `categorical` format kullanılmıştır



### 🔹 Eğitim ve Test Veri Akışı

- **Eğitim verisi (`train_gen`)**
  - Veri artırma uygulanır
  - Karıştırılarak (shuffle) modele verilir

- **Test verisi (`test_gen`)**
  - Sadece normalize edilir
  - Karıştırma yapılmaz (`shuffle=False`)
  - Bu sayede:
    - Confusion Matrix
    - ROC analizleri  
    doğru şekilde hesaplanır



### 🔹 Sınıf Bilgileri

- Toplam sınıf sayısı:
  - `num_classes = len(train_gen.class_indices)`

- Sınıf isimleri:
  - `class_names = list(train_gen.class_indices.keys())`

Bu yapı sayesinde model, veri setindeki tüm sınıfları otomatik olarak algılayabilmektedir.

---

##  Model Eğitimi: Transfer Learning ve Fine-Tuning Süreci

Bu projede birden fazla CNN modeli üzerinde aynı eğitim süreci uygulanarak performans karşılaştırması yapılmıştır.



### 🔹 Early Stopping

Modelin overfitting (aşırı öğrenme) yapmasını önlemek için Early Stopping kullanılmıştır:

- İzlenen metrik: `val_loss`
- Sabır (patience): 3 epoch
- En iyi ağırlıklar otomatik geri yüklenir (`restore_best_weights=True`)




##  1. Transfer Learning (Özellik Çıkarma Aşaması)

Her model için:

- ImageNet ağırlıkları yüklenmiştir
- Önceden eğitilmiş katmanlar dondurulmuştur (freeze)
- Bu sayede:
  - Modelin öğrendiği genel görsel özellikler korunmuştur
  - Eğitim sadece üst katmanlarda yapılmıştır



### 🔹 Model Üst Katmanı

Her modele şu yapı eklenmiştir:

- Global Average Pooling
- Fully Connected (Dense) katman (128 nöron, ReLU)
- Dropout (0.5) → overfitting önlemek için
- Çıkış katmanı (Softmax → sınıf sayısı kadar)



## ⚙️ Model Derleme

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metric: Accuracy



##  2. Fine-Tuning (İnce Ayar)

Transfer learning sonrası model daha iyi uyum sağlasın diye:

- Son 30 katman yeniden eğitilebilir hale getirilmiştir
- Daha düşük öğrenme oranı kullanılmıştır (`learning rate = 1e-5`)
- Model veri setine özel olarak yeniden optimize edilmiştir

---

## 📊 Model Değerlendirme

Her model için aşağıdaki analizler yapılmıştır:

### 🔹 Tahminler
- Test verisi üzerinde sınıf tahminleri yapılmıştır
- Olasılık değerleri (`predict_proba`) hesaplanmıştır



### 🔹 Confusion Matrix
- Gerçek ve tahmin edilen sınıflar karşılaştırılmıştır
- Sınıf bazlı performans detaylı incelenmiştir
  


### 🔹 ROC Curve & AUC
- Çok sınıflı ROC eğrileri çizilmiştir
- Her sınıf için AUC değeri hesaplanmıştır



### 🔹 Classification Report
- Precision
- Recall
- F1-score





### 🔹 Specificity (Özgüllük)

Her sınıf için ayrı ayrı hesaplanmıştır:

- True Negative oranı analiz edilmiştir
- Modelin yanlış pozitifleri ne kadar iyi ayırt ettiği ölçülmüştür

---

## 📈 Sonuçların Karşılaştırılması

Her model için şu metrikler kaydedilmiştir:

- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-score (macro average)
- Specificity (ortalama)
- AUC (macro average)

Bu metrikler karşılaştırılarak en iyi performans gösteren model belirlenmiştir.

---

##  Amaç

Bu sürecin amacı:
- Farklı CNN mimarilerini karşılaştırmak
- Transfer learning ve fine-tuning etkisini gözlemlemek
- En iyi performans veren modeli belirlemektir

