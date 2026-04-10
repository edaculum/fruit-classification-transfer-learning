
from google.colab import drive
drive.mount('/content/drive')

train_dir = "/content/drive/MyDrive/fruits/train"
test_dir = "/content/drive/MyDrive/fruits/test"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

#Data Augmentation
#Eğitim verisini yapay olarak arttırıyor

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

#Test verisi gerçek veri olmalı → değiştirilmez -> bu yüzden augmentation yok
test_datagen = ImageDataGenerator(rescale=1./255)

#Data Generator
#Görselleri klasörden okur- Batch halinde modele verir
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),  #Görseller yeniden boyutlandırılır
    batch_size=32,          #Model her adımda 32 görüntü ile eğitilir
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False            #Confusion matrix gibi analizlerde doğru sıra korunmalı
)

num_classes = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())

# Modelleri Tanımlama
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50, VGG16, InceptionV3

models_dict = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
    "ResNet50": ResNet50,
    "VGG16": VGG16,
    "InceptionV3": InceptionV3
}

results = []

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
models_predictions = {}

for model_name, model_fn in models_dict.items():

    print(f"\n Model: {model_name}")

    base_model = model_fn(weights='imagenet', include_top=False, input_shape=(224,224,3))

    #Freeze (Transfer Learning Aşaması)- Feature Extraction (özellik çıkarımı)
    #Modelin öğrenmesini DURDURUYOR- Ağırlıklar (weights) güncellenmez- Backpropagation bu katmanlara uygulanmaz- Önceden öğrenilen bilgiyi korumak için
    for layer in base_model.layers:
        layer.trainable = False

    #Üst Katmanları Ekleme
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    #Model Compile
    #Ağırlıkları günceller (sadece üst katmanlar)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #Transfer Learning- Model Eğitimi (İLK AŞAMA)
    model.fit(train_gen, validation_data=test_gen, epochs=10, verbose=1, callbacks=[early_stop])


    # FINE-TUNING
    # Unfreeze{Katmanları Aç) - dondurulan(freeze) modelin bazı katmanlarını yeniden eğitilebilir hâle getireceğiz
    #son 30 katman açılıyor
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Modeli yeniden derlerken optimizer ve learning rate belirleniyor
    # Önemli: Learning Rate Düşür (ince ayar) : Ağırlıkların güncellenme hızı
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    #  Fine-Tuning Eğitimi (10 epoch)
    # model artık son 30 katmanı güncelleyerek bizim veri setimize daha iyi uyum sağlıyor
    model.fit(train_gen, validation_data=test_gen, epochs=10, verbose=1, callbacks=[early_stop])

    #  Tahmin
    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    models_predictions[model_name] = {"y_true": y_true, "y_pred_prob": y_pred_prob}

    # ==========================
    # Confusion Matrix
    # ==========================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

    # ==========================
    # Multi-class ROC & AUC
    # ==========================
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(10,8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Multi-class ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # ==========================
    # Classification Report
    # ==========================
    print(f"\n {model_name} - Classification Report")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # ==========================
    # Specificity Sınıf Bazında
    # ==========================
    print(f"\n {model_name} - Specificity (Sınıf Bazında)")
    specificity_list = []
    for i in range(len(cm)):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP)
        specificity_list.append(specificity)
        print(f"{class_names[i]} specificity: {specificity:.4f}")

    specificity_mean = np.mean(specificity_list)

    # ==========================
    # Ortalama Metrikler
    # ==========================
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc_score = roc_auc_score(y_true_bin, y_pred_prob, average='macro', multi_class='ovr')

    results.append([model_name, acc, precision, recall, f1, specificity_mean, auc_score])

#Sonuç Tablosu
df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC"])

print("MODEL KARŞILAŞTIRMA TABLOSU")
print(df)

#EN İYİ MODEL
best_model = df.sort_values(by="Accuracy", ascending=False).iloc[0]

print("\n En iyi model:")
print(best_model)