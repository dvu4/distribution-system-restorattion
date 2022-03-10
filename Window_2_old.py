# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Window_2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets, uic
import pathlib
import time

import map_module as map_gen

current_path = os.getcwd()
data_folder = os.path.join(current_path,"ui_forms")
qtWindow_2File = os.path.join(data_folder, "Window_2.ui")

#qtWindow_2File = r"./ui_forms/Window_2.ui"
Ui_Window_2, QtBaseClass = uic.loadUiType(qtWindow_2File)


class GUI_Window_2(QtWidgets.QMainWindow, Ui_Window_2):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        Ui_Window_2.__init__(self)
        self.setupUi(self)


        ############################
        # TODO 
        ############################

        #https://www.michaelcrump.net/how-to-run-html-files-in-your-browser-from-github/
        #https://github.com/python-visualization/folium/issues/773
        #self.webEngineView.setUrl(QtCore.QUrl("http://www.google.com/"))
        #filename = "graph_polyline_r.html"
        
        filename = "output/ckt12_ieee8500_system_data.html" #intact_layer_map.html
        current_path = pathlib.Path(os.getcwd())
        full_path = current_path / filename
        self.webEngineView.setUrl(QtCore.QUrl(full_path.absolute().as_uri()))

        #filename = "http://www.anl.gov"
        #self.webEngineView.setUrl(QtCore.QUrl(filename))

        #plot networkx
        #self.pushButton.pressed.connect(self.plotNetwork)

        #open file 
        #self.pushButton_2.clicked.connect(self.openFileNameDialog)


        #quit the program
        self.pushButton_3.clicked.connect(self.closeWindow)
        #self.pushButton_3.clicked.connect(QtCore.QCoreApplication.instance().quit)
        #self.pushButton_3.clicked.connect(QtWidgets.qApp.quit)


        ######################
        self.commandLinkButton_cm1.pressed.connect(self.generateMap)
        self.commandLinkButton_dm.pressed.connect(self.showMap)


        ######################
        self.intact_grid_en     = self.radioButton
        self.est_weather_en     = self.radioButton_2
        self.est_fi_en          = self.radioButton_3
        self.est_weather_fi_en  = self.radioButton_4

        map_opt = {self.intact_grid_en     : "output/ckt12_ieee8500_system_data.html", #intact_layer_map
                   self.est_weather_en     : "output/layer_map_2.html",
                   self.est_fi_en          : "output/layer_map_3.html",
                   self.est_weather_fi_en  : "output/layer_map_4.html"} 

        self.map_opt = map_opt



    # plot network on Graphicview and rescale the image to the window
    def plotNetwork(self):
        graphicview = self.graphicsView
        width = graphicview.geometry().width()
        height = graphicview.geometry().height()

        scene = QtWidgets.QGraphicsScene()
        #scene.addPixmap(QtGui.QPixmap('networkx.png'))
        pixmap = QtGui.QPixmap('networkx.png')
        smaller_pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        scene.addPixmap(smaller_pixmap)
        graphicview.setScene(scene)
        graphicview.show()


    # https://python-forum.io/Thread-PyQt-PyQT5-Open-QFiledialog-in-a-Dialog-which-was-create-in-qt-designer?pid=16738
    def openFileNameDialog(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)


    def closeWindow(self):
        buttonReply = QtWidgets.QMessageBox.question(None, 
                                            "PyQt5 Messagebox", 
                                            "Would you like to quit this window?", 
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if buttonReply == QtWidgets.QMessageBox.Yes:
            print("Yes clicked.")
            #QtCore.QCoreApplication.instance().quit()
            import sys
            sys.exit(0)
        else:
            pass


    def showMap(self):  
        for key, value in self.map_opt.items():
            selected_radio = key
            if selected_radio.isChecked():
                filename = value

        current_path = pathlib.Path(os.getcwd())
        full_path = current_path / filename
        self.webEngineView.setUrl(QtCore.QUrl(full_path.absolute().as_uri()))
     


    def generateMap(self):
        if self.est_weather_en.isChecked():
            map_gen.gen_map_2()

        elif self.est_fi_en .isChecked():
            map_gen.gen_map_3()

        elif self.est_weather_fi_en.isChecked():
            map_gen.gen_map_4()



    #def generateMap(self):
        #map_gen.main()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui_Window_2 = GUI_Window_2()
    gui_Window_2.show()
    sys.exit(app.exec_())


#https://github.com/Programmica/pyqt5-tutorial/blob/master/_examples/radiobutton.py
#https://stackoverflow.com/questions/17402452/how-to-get-the-checked-radiobutton-from-a-groupbox-in-pyqt
#https://www.programcreek.com/python/example/68993/PyQt4.QtGui.QRadioButton
