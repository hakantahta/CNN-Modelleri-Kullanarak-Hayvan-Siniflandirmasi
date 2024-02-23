import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import  QListWidgetItem
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QSize

import os

class Ui_Dialog_IslemYapilmisVeriler(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(490, 400)
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 40, 441, 291))
        self.listWidget.setObjectName("listWidget")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(160, 10, 191, 21))
        self.label.setObjectName("label")
        yengecler = [os.path.join('./cikti/Crabs', dosya) for dosya in os.listdir('./cikti/Crabs') if dosya.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        yunuslar = [os.path.join('./cikti/Dolphin', dosya) for dosya in os.listdir('./cikti/Dolphin') if dosya.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        penguenler = [os.path.join('./cikti/Penguin', dosya) for dosya in os.listdir('./cikti/Penguin') if dosya.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        self.listWidget.clear()
        for resim in yengecler + yunuslar + penguenler:
            item = QListWidgetItem()
            item.setText(resim)
            pixmap = QPixmap(resim)

            boyut = QSize(300, 300)  
            pixmap = pixmap.scaled(boyut)  

            item.setIcon(QIcon(pixmap))
            item.setSizeHint(QSize(200, 40)) 
            self.listWidget.addItem(item)   
        

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Donem Odevi | ASP & HT"))
        self.label.setText(_translate("Dialog", "Cogaltilmis ve işlenmiş veriler"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_IslemYapilmisVeriler()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
