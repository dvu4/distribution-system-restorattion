# -*- coding: utf-8 -*-


__author__ = "Duc Vu"
__copyright__ = "Copyright 2017, " \
                "The GMLC Project: A Closed-Loop Distribution System Restoration Tool" \
                " for Natural Disaster Recovery"
__maintainer__ = "Duc Vu"
__email__ = "ducvuchicago@gmail.com"

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

#https://stackoverflow.com/questions/36852622/pyqt-window-not-closing
#https://docs.python.org/3/library/pathlib.html
#https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import pathlib
from glob import glob

from Window_1 import GUI_Window_1
from Window_2 import GUI_Window_2
from Window_3 import GUI_Window_3
from Window_4 import DSR

current_path = os.getcwd()
data_folder = os.path.join(current_path,"ui_forms")
qtMainWindowFile = os.path.join(data_folder, "MainWindow.ui")

#qtMainWindowFile = r"./ui_forms/MainWindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtMainWindowFile)


class GUI_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)


        ############################
        # TODO 
        ############################


        pic_ANL = QtGui.QPixmap("images/anl-logo.png")
        self.label_ANL.setPixmap(pic_ANL)
        self.label_ANL.setScaledContents(True)

        pic_DOE = QtGui.QPixmap("images/doe-logo.png")
        self.label_DOE.setPixmap(pic_DOE)
        self.label_DOE.setScaledContents(True)


        pic_GMLC = QtGui.QPixmap("images/logo-grid-mod-lc.png")
        self.label_GMLC.setPixmap(pic_GMLC)
        self.label_GMLC.setScaledContents(True)

        pic_grid = QtGui.QPixmap("images/distribution_grid.jpg")
        self.label_grid.setPixmap(pic_grid)
        self.label_grid.setScaledContents(True)



        self._openWindow_1  = None
        self._openWindow_2  = None
        self._openWindow_3  = None
        self._openWindow_4  = None

        #self.data_system_pickle  = None
        #self.data_weather_damage  = None
        #self.data_flood_metric  = None
        #self.data_ice_metric  = None
        #self.data_wind_metric  = None


        # show background image
        #https://forum.qt.io/topic/66311/can-t-add-a-stretched-background-image-with-pyqt5/4
        #self.setStyleSheet(" border-image: url(ANL_V_White.jpg) 0 0 0 0 stretch stretch; ")
        #self.setStyleSheet(" background-image: url(ANL_V_White.jpg) 0 0 0 0 stretch stretch; background-repeat:no-repeat;background-attachment: fixed; ")
        #open window_1
        #self.pushButton_import.clicked.connect(self.openWindow_1)
        self.commandLinkButton_import.clicked.connect(self.openWindow_1)


        #open window_2
        #self.pushButton_fault.clicked.connect(self.openWindow_2)
        self.commandLinkButton_fault.clicked.connect(self.openWindow_2)

        #open window_3
        #self.pushButton_crew.clicked.connect(self.openWindow_3)
        self.commandLinkButton_crew.clicked.connect(self.openWindow_3)

        #open window_4
        #self.pushButton_restoration.clicked.connect(self.openWindow_4)
        self.commandLinkButton_restoration.clicked.connect(self.openWindow_4)

        #quit the program
        self.pushButton_exit.clicked.connect(self.closeWindow)

        #self.delete_files



        #self.data_system_pickle = self._openWindow_1.data_system_pickle
        #self.data_weather_damage = [i for i in self._openWindow_1.data_weather_pickle if 'damge' in i][0]

        #if len(self._openWindow_1.data_weather_pickle) == 2 :
        #    self.data_flood_metric = [i for i in self._openWindow_1.data_weather_pickle if 'Flood_weather_metric' in i][0]
            
        #elif len(self._openWindow_1.data_weather_pickle) == 3:
        #    self.data_ice_metric = [i for i in self._openWindow_1.data_weather_pickle if 'Ice_weather_metric' in i][0]
        #    self.data_wind_metric = [i for i in self._openWindow_1.data_weather_pickle if 'Wind_weather_metric' in i][0]




        for filename in glob("output/layer_map_*.html"):
            if os.path.exists(filename):
                os.remove(filename)


        for filename in glob("output/*data*.html"):
            if os.path.exists(filename):
                os.remove(filename)


        for filename in glob("output/diagram_step*"):
            if os.path.exists(filename):
                os.remove(filename)


        if os.path.exists('output/data.dat'): 
            os.remove('output/data.dat')

        for filename in glob("output/*topology.png"):
            if os.path.exists(filename):
                os.remove(filename)

        #if os.path.exists('output/fault_location_data.xlsx'): 
            #os.remove('output/fault_location_data.xlsx')


    def openWindow_1(self):
        self._openWindow_1 = GUI_Window_1()
        self._openWindow_1.show()


    def openWindow_2(self):
        self._openWindow_2 = GUI_Window_2()
        if not os.path.exists('output/data.dat'): 
            buttonReply = QtWidgets.QMessageBox.question(None, 
                                            "PyQt5 Messagebox", 
                                            "Can not find data.dat !!! Please import data ", 
                                            QtWidgets.QMessageBox.Yes)
            if buttonReply == QtWidgets.QMessageBox.Yes:
                print("Yes clicked.")
                return True
        else:
            self._openWindow_2.show()



    def openWindow_3(self):
        self._openWindow_3 = GUI_Window_3()
        if not os.path.exists('output/data.dat'): 
            buttonReply = QtWidgets.QMessageBox.question(None, 
                                            "PyQt5 Messagebox", 
                                            "Can not find data.dat !!! Please import data ", 
                                            QtWidgets.QMessageBox.Yes)
            if buttonReply == QtWidgets.QMessageBox.Yes:
                print("Yes clicked.")
                return True
        else:
            self._openWindow_3.show()



    def openWindow_4(self):
        self._openWindow_4 = DSR()
        if not os.path.exists('output/data.dat'): 
            buttonReply = QtWidgets.QMessageBox.question(None, 
                                            "PyQt5 Messagebox", 
                                            "Can not find data.dat !!! Please import data ", 
                                            QtWidgets.QMessageBox.Yes)
            if buttonReply == QtWidgets.QMessageBox.Yes:
                print("Yes clicked.")
                return True
        else:
            self._openWindow_4.show()



    def closeWindow(self):
        buttonReply = QtWidgets.QMessageBox.question(None, 
                                            "PyQt5 Messagebox", 
                                            "Would you like to quit the aplication?", 
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if buttonReply == QtWidgets.QMessageBox.Yes:
            print("Yes clicked.")

            #self.delete_files
            
            for filename in glob("output/layer_map_*.html"):
                if os.path.exists(filename):
                    os.remove(filename)

            for filename in glob("output/diagram_step*"):
                if os.path.exists(filename):
                    os.remove(filename)


            if os.path.exists('output/data.dat'): 
                os.remove('output/data.dat')


            if os.path.exists('output/fault_location_data.xlsx'): 
                os.remove('output/fault_location_data.xlsx')

            for filename in glob("output/*topology.png"):
                if os.path.exists(filename):
                    os.remove(filename)
            
            sys.exit()
        else:
            pass

    '''
    def delete_files(self):
        for filename in glob("output/layer_map_*.html"):

            if os.path.exists(filename):
                os.remove(filename)

            for filename in glob("output/diagram_step*"):
                if os.path.exists(filename):
                    os.remove(filename)

            if os.path.exists('output/data.dat'): 
                os.remove('output/data.dat')
    '''


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui_MainWindow = GUI_MainWindow()
    gui_MainWindow.show()
    sys.exit(app.exec_())

    '''
    # https://python-forum.io/Thread-PyQt-PyQT5-Open-QFiledialog-in-a-Dialog-which-was-create-in-qt-designer?pid=16738
    def openFileNameDialog(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
    '''
