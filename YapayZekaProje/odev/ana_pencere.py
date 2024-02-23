from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt

from odev import calistirModelleri
from odevkfold import calistirModelleriKfold

from islem_yapilmis_veriler import Ui_Dialog_IslemYapilmisVeriler
from gercek_veriler import Ui_Dialog_Gercek_Veriler
from canlitest import canliTest

class Ui_AnaDialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(744, 676)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        Dialog.setPalette(palette)
        Dialog.setAutoFillBackground(True)
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setGeometry(QtCore.QRect(850, 30, 121, 31))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setGeometry(QtCore.QRect(850, 50, 171, 221))
        self.label_12.setObjectName("label_12")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 240, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.label_vgg16 = QtWidgets.QLabel(self.groupBox)
        self.label_vgg16.setGeometry(QtCore.QRect(0, 30, 231, 181))
        self.label_vgg16.setText("")
        self.label_vgg16.setObjectName("label_vgg16")
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(260, 240, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_2.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setCheckable(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_vgg19 = QtWidgets.QLabel(self.groupBox_2)
        self.label_vgg19.setGeometry(QtCore.QRect(-1, 25, 231, 181))
        self.label_vgg19.setText("")
        self.label_vgg19.setObjectName("label_vgg19")
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 450, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_3.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setFlat(False)
        self.groupBox_3.setCheckable(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_resnet50 = QtWidgets.QLabel(self.groupBox_3)
        self.label_resnet50.setGeometry(QtCore.QRect(-1, 25, 231, 181))
        self.label_resnet50.setText("")
        self.label_resnet50.setObjectName("label_resnet50")
        self.groupBox_4 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_4.setGeometry(QtCore.QRect(260, 450, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_4.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setFlat(False)
        self.groupBox_4.setCheckable(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_kendicnn = QtWidgets.QLabel(self.groupBox_4)
        self.label_kendicnn.setGeometry(QtCore.QRect(9, 25, 221, 181))
        self.label_kendicnn.setText("")
        self.label_kendicnn.setObjectName("label_kendicnn")
        self.groupBox_5 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 10, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_5.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setFlat(False)
        self.groupBox_5.setCheckable(False)
        self.groupBox_5.setObjectName("groupBox_5")
        self.patienceAlani = QtWidgets.QLineEdit(self.groupBox_5)
        self.patienceAlani.setGeometry(QtCore.QRect(20, 150, 191, 21))
        self.patienceAlani.setObjectName("patienceAlani")
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setGeometry(QtCore.QRect(20, 30, 60, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setGeometry(QtCore.QRect(20, 80, 71, 16))
        self.label_2.setObjectName("label_2")
        self.batchAlani = QtWidgets.QLineEdit(self.groupBox_5)
        self.batchAlani.setGeometry(QtCore.QRect(20, 100, 191, 21))
        self.batchAlani.setObjectName("batchAlani")
        self.epochAlani = QtWidgets.QLineEdit(self.groupBox_5)
        self.epochAlani.setGeometry(QtCore.QRect(20, 50, 191, 21))
        self.epochAlani.setObjectName("epochAlani")
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setGeometry(QtCore.QRect(20, 130, 141, 16))
        self.label_3.setObjectName("label_3")
        self.groupBox_6 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_6.setGeometry(QtCore.QRect(500, 10, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_6.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setFlat(False)
        self.groupBox_6.setCheckable(False)
        self.groupBox_6.setObjectName("groupBox_6")
        self.label_hibrit = QtWidgets.QLabel(self.groupBox_6)
        self.label_hibrit.setGeometry(QtCore.QRect(10, 30, 171, 151))
        self.label_hibrit.setText("")
        self.label_hibrit.setObjectName("label_hibrit")
        self.groupBox_7 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_7.setGeometry(QtCore.QRect(260, 10, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_7.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setFlat(False)
        self.groupBox_7.setCheckable(False)
        self.groupBox_7.setObjectName("groupBox_7")
        self.k_fold_yazi_alani = QtWidgets.QLineEdit(self.groupBox_7)
        self.k_fold_yazi_alani.setGeometry(QtCore.QRect(20, 50, 191, 21))
        self.k_fold_yazi_alani.setObjectName("k_fold_yazi_alani.")
        self.label_6 = QtWidgets.QLabel(self.groupBox_7)
        self.label_6.setGeometry(QtCore.QRect(20, 30, 141, 16))
        self.label_6.setObjectName("label_6")
        self.holdout_checkbox = QtWidgets.QCheckBox(self.groupBox_7)
        self.holdout_checkbox.setGeometry(QtCore.QRect(20, 80, 87, 20))
        self.holdout_checkbox.setChecked(True)
        self.holdout_checkbox.setObjectName("holdout_checkbox")
        self.kfold_checkbox = QtWidgets.QCheckBox(self.groupBox_7)
        self.kfold_checkbox.setGeometry(QtCore.QRect(120, 80, 87, 20))
        self.kfold_checkbox.setObjectName("kfold_checkbox")
        self.baslatbuton = QtWidgets.QPushButton(self.groupBox_7)
        self.baslatbuton.setGeometry(QtCore.QRect(10, 150, 211, 32))
        self.baslatbuton.setObjectName("baslatbuton")
        self.groupBox_8 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_8.setGeometry(QtCore.QRect(500, 240, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_8.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_8.setFont(font)
        self.groupBox_8.setFlat(False)
        self.groupBox_8.setCheckable(False)
        self.groupBox_8.setObjectName("groupBox_8")
        self.vgg16_buton = QtWidgets.QPushButton(self.groupBox_8)
        self.vgg16_buton.setGeometry(QtCore.QRect(10, 50, 211, 32))
        self.vgg16_buton.setObjectName("vgg16_buton")
        self.vgg19_buton = QtWidgets.QPushButton(self.groupBox_8)
        self.vgg19_buton.setGeometry(QtCore.QRect(10, 80, 211, 32))
        self.vgg19_buton.setObjectName("vgg19_buton")
        self.resnet_buton = QtWidgets.QPushButton(self.groupBox_8)
        self.resnet_buton.setGeometry(QtCore.QRect(10, 110, 211, 32))
        self.resnet_buton.setObjectName("resnet_buton")
        self.kendicnn_buton = QtWidgets.QPushButton(self.groupBox_8)
        self.kendicnn_buton.setGeometry(QtCore.QRect(10, 140, 211, 32))
        self.kendicnn_buton.setObjectName("kendicnn_buton")
        self.groupBox_9 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_9.setGeometry(QtCore.QRect(500, 450, 230, 210))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(131, 255, 10))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        self.groupBox_9.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.groupBox_9.setFont(font)
        self.groupBox_9.setFlat(False)
        self.groupBox_9.setCheckable(False)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gercekZamanTestButon = QtWidgets.QPushButton(self.groupBox_9)
        self.gercekZamanTestButon.setGeometry(QtCore.QRect(10, 110, 211, 32))
        self.gercekZamanTestButon.setObjectName("gercekZamanTestButon")
        self.cogaltilmisveributon = QtWidgets.QPushButton(self.groupBox_9)
        self.cogaltilmisveributon.setGeometry(QtCore.QRect(10, 80, 211, 32))
        self.cogaltilmisveributon.setObjectName("cogaltilmisveributon")
        self.testgosterbutton = QtWidgets.QPushButton(self.groupBox_9)
        self.testgosterbutton.setEnabled(True)
        self.testgosterbutton.setGeometry(QtCore.QRect(10, 50, 211, 32))
        self.testgosterbutton.setObjectName("testgosterbutton")
        

        self.baslatbuton.clicked.connect(self.modelleriBaslat)
        self.testgosterbutton.clicked.connect(self.gercekVerilerSayfasi)
        self.cogaltilmisveributon.clicked.connect(self.cogaltilmisVeriSayfasi)
        self.gercekZamanTestButon.clicked.connect(self.canliTestSayfasi)
        self.vgg16_buton.clicked.connect(self.gosterVGG16Grafigi)
        self.vgg19_buton.clicked.connect(self.gosterVGG19Grafigi)
        self.resnet_buton.clicked.connect(self.gosterResnet50Grafigi)
        self.kendicnn_buton.clicked.connect(self.gosterKendiCNNGrafigi)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


    train_loss = 0
    val_loss = 0
    val_loss_epoch = 0

    train_loss2 = 0
    val_loss2 = 0
    val_loss_epoch2 = 0

    train_loss3 = 0
    val_loss3 = 0
    val_loss_epoch3 = 0

    vgg16_epoch = 0
    vgg16_train_loss = 0
    vgg16_val_loss = 0
    vgg16_acc = 0
    vgg16_val_acc = 0
    
    vgg19_epoch = 0
    vgg19_train_loss = 0
    vgg19_val_loss = 0
    vgg19_acc = 0
    vgg19_val_acc = 0

    resnet50_epoch = 0
    resnet50_train_loss = 0
    resnet50_val_loss = 0
    resnet50_acc = 0
    resnet50_val_acc = 0

    kendicnn_epoch = 0
    kendicnn_train_loss = 0
    kendicnn_val_loss = 0
    kendicnn_acc = 0
    kendicnn_val_acc = 0
    
    def modelleriBaslat(self):

        modelText = 0
        modelText2 = 0
        modelText3 = 0
        modelText4 = 0
        modelText5 = 0
        train_loss = 0
        val_loss = 0
        val_loss_epoch = 0
        acc = 0
        val_acc = 0
        train_loss2 = 0
        val_loss2 = 0
        val_loss_epoch2 = 0
        acc2 = 0
        val_acc2 = 0
        train_loss3 = 0
        val_loss3 = 0
        val_loss_epoch3 = 0
        acc3 = 0
        val_acc3 = 0
        train_loss4 = 0
        val_loss4 = 0
        val_loss_epoch4 = 0
        acc4 = 0
        val_acc4 = 0

        if(self.kfold_checkbox.isChecked()):
            modelText, modelText2, modelText3, modelText4, modelText5, train_loss, val_loss, val_loss_epoch, acc, val_acc, train_loss2, val_loss2, val_loss_epoch2, acc2, val_acc2, train_loss3, val_loss3, val_loss_epoch3, acc3, val_acc3, train_loss4, val_loss4, val_loss_epoch4, acc4, val_acc4 = calistirModelleriKfold(int(self.epochAlani.text()),int(self.batchAlani.text()), int(self.patienceAlani.text()), int(self.k_fold_yazi_alani.text()))
        else:
            modelText, modelText2, modelText3, modelText4, modelText5, train_loss, val_loss, val_loss_epoch, acc, val_acc, train_loss2, val_loss2, val_loss_epoch2, acc2, val_acc2, train_loss3, val_loss3, val_loss_epoch3, acc3, val_acc3, train_loss4, val_loss4, val_loss_epoch4, acc4, val_acc4 = calistirModelleri(int(self.epochAlani.text()),int(self.batchAlani.text()), int(self.patienceAlani.text()))
            
        
        self.label_vgg16.setText(modelText)
        self.label_vgg19.setText(modelText2)
        self.label_resnet50.setText(modelText3)
        self.label_kendicnn.setText(modelText4)
        self.label_hibrit.setText(modelText5)
        self.loss_grafigi = val_loss_epoch
        self.train_loss = train_loss
        self.val_loss = val_loss

        self.vgg16_epoch = val_loss_epoch
        self.vgg16_train_loss = train_loss
        self.vgg16_val_loss = val_loss
        self.vgg16_acc = acc
        self.vgg16_val_acc = val_acc

        self.vgg19_epoch = val_loss_epoch2
        self.vgg19_train_loss = train_loss2
        self.vgg19_val_loss = val_loss2
        self.vgg19_acc = acc2
        self.vgg19_val_acc = val_acc2

        self.resnet50_epoch = val_loss_epoch3
        self.resnet50_train_loss = train_loss3
        self.resnet50_val_loss = val_loss3
        self.resnet50_acc = acc3
        self.resnet50_val_acc = val_acc3

        self.kendicnn_epoch = val_loss_epoch4
        self.kendicnn_train_loss = train_loss4
        self.kendicnn_val_loss = val_loss4
        self.kendicnn_acc = acc4
        self.kendicnn_val_acc = val_acc4
        

    def gercekVerilerSayfasi(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Dialog_Gercek_Veriler()
        self.ui.setupUi(self.window)
        self.window.show()

    def cogaltilmisVeriSayfasi(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Dialog_IslemYapilmisVeriler()
        self.ui.setupUi(self.window)
        self.window.show()
    
    def canliTestSayfasi(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = canliTest()
        self.ui.setupUi(self.window)
        self.window.show()
    
    def gosterValLossGrafigi(self):
        plt.plot(self.val_loss_epoch, self.train_loss, 'bo', label='Eğitim Loss')
        plt.plot(self.val_loss_epoch, self.val_loss, 'b', label='Validation Loss')
        plt.title('Eğitim ve Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def gosterVGG16Grafigi(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(self.vgg16_acc, label='Training Accuracy')
        ax1.plot(self.vgg16_val_acc, label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(self.vgg16_train_loss, label='Training Loss')
        ax2.plot(self.vgg16_val_loss, label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.show()
        

    def gosterVGG19Grafigi(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(self.vgg19_acc, label='Training Accuracy')
        ax1.plot(self.vgg19_val_acc, label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(self.vgg19_train_loss, label='Training Loss')
        ax2.plot(self.vgg19_val_loss, label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.show()
    
    def gosterResnet50Grafigi(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(self.resnet50_acc, label='Training Accuracy')
        ax1.plot(self.resnet50_val_acc, label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(self.resnet50_train_loss, label='Training Loss')
        ax2.plot(self.resnet50_val_loss, label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.show()

    def gosterKendiCNNGrafigi(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(self.kendicnn_acc, label='Training Accuracy')
        ax1.plot(self.kendicnn_val_acc, label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(self.kendicnn_train_loss, label='Training Loss')
        ax2.plot(self.kendicnn_val_loss, label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.show()
        

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Donem Odevi | ASP & HT"))
        self.label_11.setText(_translate("Dialog", "CNN"))
        self.label_12.setText(_translate("Dialog", "Kendi CNN Sonuc"))
        self.groupBox.setTitle(_translate("Dialog", "VGG16"))
        self.groupBox_2.setTitle(_translate("Dialog", "VGG19"))
        self.groupBox_3.setTitle(_translate("Dialog", "ResNet50"))
        self.groupBox_4.setTitle(_translate("Dialog", "Kendi CNN"))
        self.groupBox_5.setTitle(_translate("Dialog", "Özellikler"))
        self.patienceAlani.setText(_translate("Dialog", "3"))
        self.label.setText(_translate("Dialog", "Epoch"))
        self.label_2.setText(_translate("Dialog", "Batch Size"))
        self.batchAlani.setText(_translate("Dialog", "64"))
        self.epochAlani.setText(_translate("Dialog", "3"))
        self.label_3.setText(_translate("Dialog", "Sabir Degeri"))
        self.groupBox_6.setTitle(_translate("Dialog", "Hibrit Model"))
        self.groupBox_7.setTitle(_translate("Dialog", "K-Fold - Hold Out"))
        self.k_fold_yazi_alani.setText(_translate("Dialog", "2"))
        self.label_6.setText(_translate("Dialog", "K-Fold Degeri"))
        self.holdout_checkbox.setText(_translate("Dialog", "Hold Out"))
        self.kfold_checkbox.setText(_translate("Dialog", "K-Fold"))
        self.baslatbuton.setText(_translate("Dialog", "Başlat"))
        self.groupBox_8.setTitle(_translate("Dialog", "Grafikler"))
        self.vgg16_buton.setText(_translate("Dialog", "VGG16"))
        self.vgg19_buton.setText(_translate("Dialog", "VGG19"))
        self.resnet_buton.setText(_translate("Dialog", "ResNET50"))
        self.kendicnn_buton.setText(_translate("Dialog", "Kendi CNN Modelimiz"))
        self.groupBox_9.setTitle(_translate("Dialog", "Islemler"))
        self.gercekZamanTestButon.setText(_translate("Dialog", "Gerçek Zamanlı Test Sistemi"))
        self.cogaltilmisveributon.setText(_translate("Dialog", "Cogaltilmis Veriler"))
        self.testgosterbutton.setText(_translate("Dialog", "Test Verisi"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_AnaDialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
