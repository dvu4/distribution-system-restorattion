# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Window_3.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import pathlib
from time import strftime, localtime

current_path = os.getcwd()
data_folder = os.path.join(current_path,"ui_forms")
qtWindow_3File = os.path.join(data_folder, "Window_3.ui")

#qtWindow_3File = r"./ui_forms/Window_3.ui"
Ui_Window_3, QtBaseClass = uic.loadUiType(qtWindow_3File)


class GUI_Window_3(QtWidgets.QMainWindow, Ui_Window_3):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        Ui_Window_3.__init__(self)
        self.setupUi(self)

        ############################
        # TODO 
        ############################

        # Display the GIS map
        #filename = "graph_polyline_r.html"
        filename = "output/intact_layer_map.html"
        current_path = pathlib.Path(os.getcwd())
        full_path = current_path / filename
        self.webEngineView.setUrl(QtCore.QUrl(full_path.absolute().as_uri()))

        #plot networkx
        #self.pushButton.pressed.connect(self.plotNetwork)



        #open file 
        #self.pushButton_2.clicked.connect(self.openFileNameDialog)


        #quit the program
        self.pushButton_exit.clicked.connect(self.closeWindow)

        message = 'Damaged lines:'
        self.textBrowser_input.setText(message)
        message = 'Depot index:'
        self.textBrowser_input.append(message)





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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui_Window_3 = GUI_Window_3()
    gui_Window_3.show()
    sys.exit(app.exec_())