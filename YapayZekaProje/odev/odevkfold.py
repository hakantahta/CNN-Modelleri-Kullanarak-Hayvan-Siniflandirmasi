import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow import keras
import logging
logging.getLogger('tensorflow').disabled = True
from keras.layers import Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras import layers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tqdm import tqdm
import os
from PIL import Image
from tensorflow.keras.layers import Input, Concatenate
import pickle
from sklearn.preprocessing import LabelBinarizer

# Hybrid model için tahminlerin ortalama değerini hesaplamak için bir fonksiyon tanımlıyoruz.
def olustur_hybrid_model(model, model2, model3, model4, x_test_resimler, y_kat_test):
        modeller = [model, model2, model3, model4]
        tahminler = np.array([model.predict(x_test_resimler) for model in modeller])
        ortalama_tahminler = np.mean(tahminler, axis=0)
        hybrid_tahminler = np.argmax(ortalama_tahminler, axis=1)
        return accuracy_score(np.argmax(y_kat_test, axis=1), hybrid_tahminler)

def calistirModelleriKfold(epochs, batchSize, patience, k_folds):
    resimler = []
    labeller = []

    ana_dizin = './girdiler'

    # Girdi dizinindeki her bir sınıf için veri yükleme
    for hayvan in tqdm(os.listdir(ana_dizin)):
        gecerli_dizin = os.path.join(ana_dizin, hayvan)
        
        if os.path.isdir(gecerli_dizin):
            for i in range(len(os.listdir(gecerli_dizin))):
                resim_yolu = os.path.join(gecerli_dizin, os.listdir(gecerli_dizin)[i])
                resim = cv2.imread(resim_yolu)
                resim_yeniden_boyutlandir = cv2.resize(resim, (229, 229))
                resim_yeniden_boyutlandir = resim_yeniden_boyutlandir / 255
                resimler.append(resim_yeniden_boyutlandir)
                labeller.append(hayvan)

    # Veriyi numpy dizisine dönüştürme
    resimler = np.array(resimler, dtype='float32')

    # Etiketlerin sayısallaştırılması
    le = preprocessing.LabelEncoder()
    le.fit(labeller)
    class_isimleri = le.classes_
    labeller = le.transform(labeller)

    labeller = np.array(labeller, dtype='uint8')
    labeller = np.resize(labeller, (len(labeller), 1))

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    dogruluklar = []

    for fold, (train_index, test_index) in enumerate(kfold.split(resimler, labeller)):
        x_train, x_test = resimler[train_index], resimler[test_index]
        y_train, y_test = labeller[train_index], labeller[test_index]

        y_kat_train = to_categorical(y_train, len(class_isimleri))
        y_kat_test = to_categorical(y_test, len(class_isimleri))

        dataolustur = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )

        coklanmis_resimler = []
        coklanmis_labeller = []

        for i in range(len(x_train)):
            resim = x_train[i]
            label = y_kat_train[i]

            resim = np.reshape(resim, (1,) + resim.shape)

            for batch in dataolustur.flow(resim, batch_size= batchSize):
                coklanmis_resimler.append(batch[0])
                coklanmis_labeller.append(label)
                break

        # Veri artırılmış veriyi numpy dizisine dönüştürme
        coklanmis_resimler = np.array(coklanmis_resimler)
        coklanmis_labeller = np.array(coklanmis_labeller)

        # Eğitim verisini birleştirme
        x_train_coklanmis_resimler = np.concatenate((x_train, coklanmis_resimler))
        y_cat_train_coklanmis = np.concatenate((y_kat_train, coklanmis_labeller))

        classlar_bilgiler = {}
        classlar = sorted(os.listdir(ana_dizin))

        for isim in classlar:
            class_yolu = os.path.join(ana_dizin, isim)

            if os.path.isdir(class_yolu):
                classlar_bilgiler[isim] = len(os.listdir(class_yolu))


        early_stop = EarlyStopping(monitor='val_loss', patience=patience)

        # VGG16 Modeli
        model_path = f"model_{fold + 1}.h5"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        tf.config.run_functions_eagerly(True)

        vgg16_model = VGG16(weights='imagenet', input_shape=(229, 229, 3), include_top=False)

        x_train_resimler_boyutlandirildi = np.array([cv2.resize(img, (229, 229)) for img in x_train_coklanmis_resimler])

        vgg16_model.trainable = False
        global_average_katmani = GlobalAveragePooling2D()(vgg16_model.output)
        tahmin_katmani = Dense(len(class_isimleri), activation='softmax')(global_average_katmani)
        model = Model(inputs=vgg16_model.input, outputs=tahmin_katmani)

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x_train_resimler_boyutlandirildi, y_cat_train_coklanmis, epochs=epochs,
                  validation_data=(x_test, y_kat_test),
                  batch_size=batchSize,
                  callbacks=[early_stop, model_checkpoint_callback])

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        val_loss_epoch = range(1, len(train_loss) + 1)

        model.load_weights(model_path)

        # Test verisi üzerinde tahmin yapma
        y_tahmin = model.predict(x_test)
        y_tahmin_classlari = np.argmax(y_tahmin, axis=1)

        y_true = np.argmax(y_kat_test, axis=1)

        # Sınıflandırma raporunu oluşturma ve değerlendirme
        class_sonuc_raporu = classification_report(y_true, y_tahmin_classlari, target_names=class_isimleri, output_dict=True)
        basarim = class_sonuc_raporu['accuracy']
        # Başarı ve diğer metrikleri yazdırma
        vgg16ModelYazi = "Başarım: " + str(round(basarim, 2)) + "\n" + "Hassasiyet (Yengeç): " + str(round(class_sonuc_raporu["Crabs"]['recall'], 2)) + "\n" + "Hassasiyet (Yunus): " + str(round(class_sonuc_raporu["Dolphin"]['recall'], 2)) + "\n" + "Hassasiyet (Penguen): " + str(round(class_sonuc_raporu["Penguin"]['recall'],2)) + "\n" + "Özgüllük (Yengeç): " + str(round(class_sonuc_raporu["Crabs"]['precision'], 2)) + "\n" + "Özgüllük (Yunus): " + str(round(class_sonuc_raporu["Dolphin"]['precision'], 2)) + "\n" + "Özgüllük (Penguen): " + str(round(class_sonuc_raporu["Penguin"]['precision'], 2))

        dogruluklar.append(basarim)

        # VGG19 Modeli
        model_path = f"model2_{fold + 1}.h5"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        tf.config.run_functions_eagerly(True)

        vgg19_model = VGG19(weights='imagenet', input_shape=(229, 229, 3), include_top=False)

        vgg19_model.trainable = False
        global_average_katmani2 = GlobalAveragePooling2D()(vgg19_model.output)
        tahmin_katmani2 = Dense(len(class_isimleri), activation='softmax')(global_average_katmani2)
        model2 = Model(inputs=vgg19_model.input, outputs=tahmin_katmani2)

        model2.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        history2 = model2.fit(x_train_resimler_boyutlandirildi, y_cat_train_coklanmis, epochs=epochs,
                  validation_data=(x_test, y_kat_test),
                  batch_size=batchSize,
                  callbacks=[early_stop, model_checkpoint_callback])

        train_loss2 = history2.history['loss']
        val_loss2 = history2.history['val_loss']
        acc2 = history2.history['accuracy']
        val_acc2 = history2.history['val_accuracy']
        val_loss_epoch2 = range(1, len(train_loss2) + 1)

        model2.load_weights(model_path)
        
        y_tahmin2 = model.predict(x_test)
        y_tahmin_classlari2 = np.argmax(y_tahmin2, axis=1)

        y_true = np.argmax(y_kat_test, axis=1)

        class_sonuc_raporu2 = classification_report(y_true, y_tahmin_classlari2, target_names=class_isimleri, output_dict=True)
        basarim2 = class_sonuc_raporu2['accuracy']
        vgg19ModelYazi = "Başarım: " + str(round(basarim2, 2)) + "\n" + "Hassasiyet (Yengeç): " + str(round(class_sonuc_raporu2["Crabs"]['recall'], 2)) + "\n" + "Hassasiyet (Yunus): " + str(round(class_sonuc_raporu2["Dolphin"]['recall'], 2)) + "\n" + "Hassasiyet (Penguen): " + str(round(class_sonuc_raporu2["Penguin"]['recall'],2)) + "\n" + "Özgüllük (Yengeç): " + str(round(class_sonuc_raporu2["Crabs"]['precision'], 2)) + "\n" + "Özgüllük (Yunus): " + str(round(class_sonuc_raporu2["Dolphin"]['precision'], 2)) + "\n" + "Özgüllük (Penguen): " + str(round(class_sonuc_raporu2["Penguin"]['precision'], 2))
    
        dogruluklar.append(basarim2)

        # ResNet50 Modeli
        model_path = f"model3_{fold + 1}.h5"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        tf.config.run_functions_eagerly(True)

        resNet_Model = ResNet50(weights='imagenet', input_shape=(229, 229, 3), include_top=False)

        resNet_Model.trainable = False
        global_average_katmani3 = GlobalAveragePooling2D()(resNet_Model.output)
        tahmin_katmani3 = Dense(len(class_isimleri), activation='softmax')(global_average_katmani3)
        model3 = Model(inputs=resNet_Model.input, outputs=tahmin_katmani3)

        model3.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        history3 = model3.fit(x_train_resimler_boyutlandirildi, y_cat_train_coklanmis, epochs=epochs,
                  validation_data=(x_test, y_kat_test),
                  batch_size=batchSize,
                  callbacks=[early_stop, model_checkpoint_callback])
        
        train_loss3 = history3.history['loss']
        val_loss3 = history3.history['val_loss']
        acc3 = history3.history['accuracy']
        val_acc3 = history3.history['val_accuracy']

        val_loss_epoch3 = range(1, len(train_loss3) + 1)

        model3.load_weights(model_path)

        y_tahmin3 = model3.predict(x_test)
        y_tahmin_classlari3 = np.argmax(y_tahmin3, axis=1)

        y_true = np.argmax(y_kat_test, axis=1)

        class_sonuc_raporu3 = classification_report(y_true, y_tahmin_classlari3, target_names=class_isimleri, output_dict=True)
        basarim3 = class_sonuc_raporu3['accuracy']
        resNetModelYazi = "Başarım: " + str(round(basarim3, 2)) + "\n" + "Hassasiyet (Yengeç): " + str(round(class_sonuc_raporu3["Crabs"]['recall'], 2)) + "\n" + "Hassasiyet (Yunus): " + str(round(class_sonuc_raporu3["Dolphin"]['recall'], 2)) + "\n" + "Hassasiyet (Penguen): " + str(round(class_sonuc_raporu3["Penguin"]['recall'],2)) + "\n" + "Özgüllük (Yengeç): " + str(round(class_sonuc_raporu3["Crabs"]['precision'], 2)) + "\n" + "Özgüllük (Yunus): " + str(round(class_sonuc_raporu3["Dolphin"]['precision'], 2)) + "\n" + "Özgüllük (Penguen): " + str(round(class_sonuc_raporu3["Penguin"]['precision'], 2))
    
        dogruluklar.append(basarim3)

        model4_yolu = f"model4_{fold + 1}.h5"
        model_checkpoint_callback4 = ModelCheckpoint(
            filepath=model4_yolu,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        tf.config.run_functions_eagerly(True)

        model4 = Sequential()
        model4.add(Conv2D(32, (11, 11), activation='relu', kernel_initializer='he_uniform', padding='same',
                        input_shape=(229, 229, 3)))
        model4.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model4.add(MaxPooling2D((2, 2)))
        model4.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model4.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model4.add(MaxPooling2D((2, 2)))
        model4.add(Flatten())
        model4.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model4.add(Dense(units=len(class_isimleri), activation='softmax')) 

        model4.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        history4= model4.fit(x_train_resimler_boyutlandirildi, y_cat_train_coklanmis, epochs=epochs,
                        validation_data=(x_test, y_kat_test),
                        batch_size=batchSize,
                        callbacks=[early_stop, model_checkpoint_callback4])
        
        train_loss4 = history4.history['loss']
        val_loss4 = history4.history['val_loss']
        acc4 = history4.history['accuracy']
        val_acc4 = history4.history['val_accuracy']

        val_loss_epoch4 = range(1, len(train_loss4) + 1)

        model4.load_weights(model4_yolu)

        y_tahmin4 = model4.predict(x_test)
        y_tahmin_classlari4 = np.argmax(y_tahmin4, axis=1)

        y_true = np.argmax(y_kat_test, axis=1)

        class_sonuc_raporu4 = classification_report(y_true, y_tahmin_classlari4, target_names=class_isimleri, output_dict=True)
        basarim4 = class_sonuc_raporu4['accuracy']
        kendiCnnYazi = "Başarım: " + str(round(basarim4, 2)) + "\n" + "Hassasiyet (Yengeç): " + str(round(class_sonuc_raporu4["Crabs"]['recall'], 2)) + "\n" + "Hassasiyet (Yunus): " + str(round(class_sonuc_raporu4["Dolphin"]['recall'], 2)) + "\n" + "Hassasiyet (Penguen): " + str(round(class_sonuc_raporu4["Penguin"]['recall'],2)) + "\n" + "Özgüllük (Yengeç): " + str(round(class_sonuc_raporu4["Crabs"]['precision'], 2)) + "\n" + "Özgüllük (Yunus): " + str(round(class_sonuc_raporu4["Dolphin"]['precision'], 2)) + "\n" + "Özgüllük (Penguen): " + str(round(class_sonuc_raporu4["Penguin"]['precision'], 2))   

        dogruluklar.append(basarim4)


    model_text_hybrid = olustur_hybrid_model(model, model2, model3, model4, x_test, y_kat_test)
    hybridModelYazi = "Başarım: " + str(round(model_text_hybrid, 2))

    return vgg16ModelYazi, vgg19ModelYazi, resNetModelYazi, kendiCnnYazi, hybridModelYazi, train_loss, val_loss, val_loss_epoch, acc, val_acc, train_loss2, val_loss2, val_loss_epoch2, acc2, val_acc2, train_loss3, val_loss3, val_loss_epoch3, acc3, val_acc3,train_loss4, val_loss4, val_loss_epoch4, acc4, val_acc4


