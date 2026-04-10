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

---

### 🔹 Data Generator Yapısı

Keras `ImageDataGenerator` yapısı kullanılarak görseller klasörlerden okunmuş ve model eğitimine uygun hale getirilmiştir.

- Görseller `224x224` boyutuna yeniden ölçeklendirilmiştir
- Veriler batch (grup) halinde modele verilmiştir (`batch_size=32`)
- Çok sınıflı sınıflandırma için `categorical` format kullanılmıştır

---

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

---

### 🔹 Sınıf Bilgileri

- Toplam sınıf sayısı:
  - `num_classes = len(train_gen.class_indices)`

- Sınıf isimleri:
  - `class_names = list(train_gen.class_indices.keys())`

Bu yapı sayesinde model, veri setindeki tüm sınıfları otomatik olarak algılayabilmektedir.

---

##  Eğitim Süreci

### 1. Transfer Learning
- Önceden eğitilmiş model yüklenir
- Tüm katmanlar dondurulur (freeze)
- Sadece üst katmanlar eğitilir

### 2. Fine-Tuning
- Son 30 katman açılır
- Düşük learning rate ile yeniden eğitilir

---

## 📊 Değerlendirme Metrikleri

Her model aşağıdaki metriklere göre değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- AUC (ROC)

---

## Görselleştirmeler

Her model için:

- Confusion Matrix
- ROC Curve (çok sınıflı)
- Classification Report

---

## Sonuç

Tüm modeller karşılaştırılarak en yüksek doğruluğa (Accuracy) sahip model belirlenmiştir.

---


