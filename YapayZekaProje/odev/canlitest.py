import pickle
import tkinter as tk
import cv2
import numpy as np
import os

from tkinter import filedialog
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

def canliTest():
    resimler = []
    labeller = []
    ana_dizin = './girdiler'

    for hayvan in tqdm(os.listdir(ana_dizin)):
        gecerli_dizin = os.path.join(ana_dizin, hayvan)

        if os.path.isdir(gecerli_dizin):
            for i in range(len(os.listdir(gecerli_dizin))):
                resim_yolu = os.path.join(gecerli_dizin, os.listdir(gecerli_dizin)[i])
                resim = cv2.imread(resim_yolu)
                resim_yeniden_boyutlandir = cv2.resize(resim, (229, 229))
                resim_yeniden_boyutlandir = resim_yeniden_boyutlandir / 255.0
                resimler.append(resim_yeniden_boyutlandir)
                labeller.append(hayvan)

    resimler = np.array(resimler, dtype='float32')

    le = preprocessing.LabelEncoder()
    le.fit(labeller)
    class_isimleri = le.classes_
    labeller = le.transform(labeller)

    lb = LabelBinarizer()
    labeller = lb.fit_transform(labeller)

    with open("lb.pickle", "wb") as f:
        f.write(pickle.dumps(lb))

    kok = tk.Tk()
    kok.title("Gerçek Zamanlı Test")

    pencere_genisligi = 600
    pencere_yuksekligi = 400
    
    ekran_genisligi = kok.winfo_screenwidth()
    ekran_yuksekligi = kok.winfo_screenheight()
    x_kordinati = (ekran_genisligi - pencere_genisligi) / 2
    y_kordinati = (ekran_yuksekligi - pencere_yuksekligi) / 2
    kok.geometry(f"{pencere_genisligi}x{pencere_yuksekligi}+{int(x_kordinati)}+{int(y_kordinati)}")
    liste_kutusu = tk.Listbox(kok, selectmode=tk.SINGLE, width=100, height=10)
    liste_kutusu.pack(pady=10)

    def dosya_sec():
        dosya_yolu = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if dosya_yolu:
            liste_kutusu.insert(tk.END, dosya_yolu)

    
    sec_buton = tk.Button(kok, text="Dosya Seç", command=dosya_sec)
    sec_buton.pack(pady=10)
    
    model = load_model("model.h5")
    def resmi_tahmin_et():
        secilen_dosya = liste_kutusu.get(tk.ACTIVE)

        resim = load_img(secilen_dosya, target_size=(229, 229))
        resim = img_to_array(resim) / 255.0
        resim = np.expand_dims(resim, axis=0)

        tahminler = model.predict(resim)
        tahmin_class_indexi = np.argmax(tahminler, axis=1)[0]

        tahmin_sinifi = class_isimleri[tahmin_class_indexi]

        sonuc_label.config(text=f"Görüntü Tahmin Edilen Sınıf: {tahmin_sinifi}")

    tahmin_butonu = tk.Button(kok, text="Tahmin Yap", command=resmi_tahmin_et)
    tahmin_butonu.pack(pady=10)

    sonuc_label = tk.Label(kok, text="")
    sonuc_label.pack(pady=10)

    kok.mainloop()

