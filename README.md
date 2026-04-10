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



## ⚙️ Veri Ön İşleme

- Görseller `224x224` boyutuna yeniden boyutlandırılmıştır
- Normalize edilmiştir (`rescale=1./255`)
- Eğitim verisine:
  - Döndürme (rotation)
  - Zoom
  - Yatay çevirme (flip)

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

## Kullanılan Kütüphaneler

```bash
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
