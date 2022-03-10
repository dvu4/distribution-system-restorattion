# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Window_1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets, uic
import pathlib

import pickle
import pandas as pd
import numpy as np
import re
import glob

from jinja2 import Template

from data_import_module import *
from restoration_module import *
from plot_dss_module import *

import flood_map_module as flood_map
import storm_map_module as storm_map


current_path = os.getcwd()
data_folder = os.path.join(current_path,"ui_forms")
qtWindow_1File = os.path.join(data_folder, "Window_1.ui")

#qtWindow_1File = r"./ui_forms/Window_1.ui"
Ui_Window_1, QtBaseClass = uic.loadUiType(qtWindow_1File)



class GUI_Window_1(QtWidgets.QMainWindow, Ui_Window_1):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        Ui_Window_1.__init__(self)
        self.setupUi(self)

        ############################
        # TODO 
        ############################

        #https://www.michaelcrump.net/how-to-run-html-files-in-your-browser-from-github/
        #https://github.com/python-visualization/folium/issues/773
        #self.webEngineView.setUrl(QtCore.QUrl("http://www.google.com/"))
        #filename = "graph_polyline.html"
        #current_path = pathlib.Path(os.getcwd())
        #full_path = current_path / filename
        #print(current_path)
        #print('file://' + current_path + '/' + filename)
        #self.webEngineView.setUrl(QtCore.QUrl(full_path.absolute().as_uri()))
        #self.webEngineView.setUrl(QtCore.QUrl('file://' + current_path + '/' + filename))
        #self.webEngineView.setUrl(QtCore.QUrl("file:///C:/Chen/Dropbox/coding/GUI_Qt5_6_v3/GUI_Qt5_6_v3/graph_polyline.html"))
        #self.webEngineView.setUrl(QtCore.QUrl("file:///Users/ducvu/Desktop/gui_qt_webview/graph_polyline.html"))
        
        #current_path = os.getcwd()
        #dssFileList = os.listdir(current_path + "/test_system_data/")

        current_path = os.getcwd()
        #dssFileList = glob.glob(current_path + "/test_system_data/" + "*.dat")
        #dssFileList = [re.sub(current_path + "/test_system_data/",'',i) for i in dssFileList]


        system_option = {'IEEE123'              : './data/test_system_data/IEEE123/',
                         '10K node system'      : './data/test_system_data/IEEE8500/' ,
                         'User defined system'  : './data/test_system_data/user-defined-system'} 

        self.system_option = system_option
        systemFolderList = list(system_option.keys())


        weather_option = {'FLOOD'      : './data/weather/FLOOD/',
                          'STORM'      : './data/weather/STORM/' } 


        self.weather_option = weather_option
        weatherFolderList = list(weather_option.keys())
     


        #systemFolderList = glob.glob(current_path + '/data/test_system_data/*/') 
        #weatherFolderList = glob.glob(current_path + '/data/test_system_data/weather/*/FLOOD/') 

        dssFolderList = glob.glob(current_path + "/test_system_data/*/")
        self.data_pickle = None
        self.data_system_pickle = None
        self.data_weather_pickle = []
        
        self.fileNameNode = None
        self.fileNameLine = None
        self.fileNameSwitch = None
        self.fileNameRegulator = None
        self.fileNameLoad = None
        self.fileNameCapacitor = None
        self.fileNameDG = None
        self.fileNameESS = None

        self.fileNameDSS = None

        
        
        self.toolButton_node.pressed.connect(self.openFileNameDialogNode)
        self.toolButton_line.pressed.connect(self.openFileNameDialogLine)
        self.toolButton_switch.pressed.connect(self.openFileNameDialogSwitch)
        self.toolButton_regulator.pressed.connect(self.openFileNameDialogRegulator)
        self.toolButton_load.pressed.connect(self.openFileNameDialogLoad)
        self.toolButton_capacitor.pressed.connect(self.openFileNameDialogCapacitor)
        self.toolButton_dg.pressed.connect(self.openFileNameDialogDG)    
        self.toolButton_ess.pressed.connect(self.openFileNameDialogESS)
        
        #plot networkx
        #self.pushButton.pressed.connect(self.plotNetwork)


        #open file 
        #self.pushButton_2.clicked.connect(self.openFileNameDialog)


        #quit the program
        self.pushButton_3.clicked.connect(self.closeWindow)
        #self.pushButton_3.clicked.connect(QtCore.QCoreApplication.instance().quit)
        #self.pushButton_3.clicked.connect(QtWidgets.qApp.quit)


        #wordlist = ['correct1', 'correct2', 'incorrect1', 'correct3', 'incorrect2']
        #self.textBrowser.append('incorrect1')
        #wordlist = self.read_pickle(data_pickle)
        #wordlist = self.openFeederModel
        #wordlist = os.path.abspath(wordlist)
        # cursor = self.textBrowser.textCursor()
        # cursor.insertHtml('''<p><span style="color: red;">{} </span>'''.format(wordlist))
        #self.lineEdit_2.setText(str(f1))
        #self.lineEdit_2.textChanged.connect(f1)


        self.buttonSummary.clicked.connect(self.summary_click)
        self.buttonDefault.clicked.connect(self.default_click)



        # mutually exculsive checkboxes 
        #self.groupBox_dss.setEnabled(False)
        #self.radioButton_sm.setChecked(True)
        ##self.groupBox.setEnabled(True)
        ##self.radioButton_dss.setChecked(True)



        self.groupBox_123.setEnabled(True)
        self.radioButton_sm.setChecked(True)
        #self.groupBox.setEnabled(True)
        #self.radioButton_123.setChecked(True)
        

        
        self.radioButton_sm.toggled.connect(lambda:self.btnstate(self.radioButton_sm))
        #self.radioButton_dss.toggled.connect(lambda:self.btnstate(self.radioButton_dss))
        self.radioButton_123.toggled.connect(lambda:self.btnstate(self.radioButton_123))

        #self.comboBox_dss.addItem("IEEE8500_system_data")
        #self.comboBox_dss.addItem("ckt5_system_data")
        #self.comboBox_dss.addItem("ckt7_system_data")
        #self.comboBox_dss.addItem("ckt24_system_data")
        #self.comboBox_dss.addItems(dssFileList)
        self.comboBox_dss.addItems(dssFolderList)
        self.comboBox_dss.activated[str].connect(self.onActivated)    


        self.comboBox_system.addItems(systemFolderList)
        self.comboBox_system.activated[str].connect(self.onActivatedSystem)    


        self.comboBox_weather_hazard.addItems(weatherFolderList)
        self.comboBox_weather_hazard.activated[str].connect(self.onActivatedWeather)    



        self.buttonPlot.clicked.connect(self.plot_dss_networks)
        self.buttonSummary2.clicked.connect(self.summary_click_dss)
        self.toolButton_dss.pressed.connect(self.openFileNameDialogDSS)




        
        self.data_system_pickle  = None
        self.data_weather_damage  = None
        self.data_flood_metric  = None
        self.data_ice_metric  = None
        self.data_wind_metric  = None


    
    
    def retrieve_data(self):
        return self.data_system_pickle, self.data_weather_pickle


    def plot_dss_networks(self):
        topologyFile, topologyFilePath  = plot_topological_distribution_networks(self.data_pickle)
        self.popupWindow = popupWindow(topologyFile, topologyFilePath)
        self.popupWindow.show()


    def onActivated(self, filePath):
        fileNames = glob.glob(filePath + "*.dat")
        self.data_pickle =  fileNames[0]
        print(self.data_pickle)


    def generateMap(self):
        
        self.data_weather_damage = [i for i in self.data_weather_pickle if 'damge' in i][0]

        if len(self.data_weather_pickle) == 2 :
            self.data_flood_metric = [i for i in self.data_weather_pickle if 'Flood_weather_metric' in i][0]
            flood_map.gen_flood_map(self.data_system_pickle, self.data_flood_metric, self.data_weather_damage)
            
        elif len(self.data_weather_pickle) == 3:
            self.data_ice_metric = [i for i in self.data_weather_pickle if 'Ice_weather_metric' in i][0]
            self.data_wind_metric = [i for i in self.data_weather_pickle if 'Wind_weather_metric' in i][0]
            storm_map.gen_storm_map(self.data_system_pickle, self.data_ice_metric, self.data_wind_metric, self.data_weather_damage)



    def onActivatedSystem(self, filePath):
        fileNames = glob.glob(self.system_option[filePath] + "*.dat")
        if len(fileNames):
            self.data_system_pickle =  fileNames[0]
            print(self.data_system_pickle)
        else:
            print('No file was found')



    def onActivatedWeather(self, filePath):
        system_name = self.data_system_pickle.split('/')[-2]
        fileNames = glob.glob(self.weather_option[filePath] + str(system_name) + "/*.xlsx")
        self.data_weather_pickle =  fileNames
        print(self.data_weather_pickle)

        self.generateMap()




    def btnstate(self,b):
        if b.text() == "Load DSS model":
            if b.isChecked() == True:
                self.groupBox_dss.setEnabled(True)
            else:
                self.groupBox_dss.setEnabled(False)


        if b.text() == "Load System Model and Parameters":
            if b.isChecked() == True:
                self.groupBox.setEnabled(True)
            else:
                self.groupBox.setEnabled(False)


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

    def openFeederModel(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileName_feeder, _ = QtWidgets.QFileDialog.getOpenFileName(None,
                                                                        "Load System Feeder Data","",
        
                                                                     "All Files (*);;Excel Files (*.xls)", options=options)
        self.write_pickle(os.path.abspath(self.fileName_feeder))

        if self.fileName_feeder:
            print(self.fileName_feeder)
        


    def openFileNameDialog(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        #fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if fileName:
            print(fileName)



    #https://www.programcreek.com/python/example/90843/PyQt5.QtWidgets.QMessageBox.Yes
    def openFileNameDialogNode(self): 
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameNode, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameNode:
            print(self.fileNameNode)

        while self.checkLegitFile(self.fileNameNode, 'node'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameNode, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
    
        self.lineEdit_node.setText(self.fileNameNode)

  


    def openFileNameDialogLine(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameLine, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameLine:
            print(self.fileNameLine)

        while self.checkLegitFile(self.fileNameLine, 'line'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameLine, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_line.setText(self.fileNameLine)




    def openFileNameDialogSwitch(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameSwitch, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameSwitch:
            print(self.fileNameSwitch)

        while self.checkLegitFile(self.fileNameSwitch, 'switch'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameSwitch, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_switch.setText(self.fileNameSwitch)



    def openFileNameDialogRegulator(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameRegulator, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameRegulator:
            print(self.fileNameRegulator)

        while self.checkLegitFile(self.fileNameRegulator, 'regulator'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameRegulator, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_regulator.setText(self.fileNameRegulator)



    def openFileNameDialogLoad(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameLoad, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameLoad:
            print(self.fileNameLoad)

        while self.checkLegitFile(self.fileNameLoad, 'loads'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameLoad, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_load.setText(self.fileNameLoad)



    def openFileNameDialogCapacitor(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameCapacitor, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameCapacitor:
            print(self.fileNameCapacitor)

        while self.checkLegitFile(self.fileNameCapacitor, 'cap'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameCapacitor, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_capacitor.setText(self.fileNameCapacitor)



    def openFileNameDialogDG(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameDG, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameDG:
            print(self.fileNameDG)

        while self.checkLegitFile(self.fileNameDG, 'dg'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameDG, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_dg.setText(self.fileNameDG)



    def openFileNameDialogESS(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameESS, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)
        if self.fileNameESS:
            print(self.fileNameESS)

        while self.checkLegitFile(self.fileNameESS, 'ess'):   
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.fileNameESS, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Excel Files (*.xls)", options=options)

        self.lineEdit_ess.setText(self.fileNameESS)


    def checkLegitFile(self,fileName, keyword):
        #lowercase all words and split them in original fileName
        keywords = re.split(r'[`\s\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', fileName.lower())

        #check in keyword if it exists in fileName 
        if keyword not in keywords:
            buttonReply = QtWidgets.QMessageBox.question(None, 
                                            "PyQt5 Messagebox", 
                                            "Incorrect file !!! Please choose the " + keyword + " file?", 
                                            QtWidgets.QMessageBox.Yes)
            if buttonReply == QtWidgets.QMessageBox.Yes:
                print("Yes clicked.")
                return True
            #else:
                #pass
                




    def summary_click(self):
        #self.delete_pickle()

        data_read(fileNameNode = self.fileNameNode, fileNameLine = self.fileNameLine, 
                fileNameSwitch = self.fileNameSwitch, fileNameRegulator = self.fileNameRegulator, 
                fileNameLoad = self.fileNameLoad, fileNameCapacitor = self.fileNameCapacitor, 
                fileNameDG = self.fileNameDG, fileNameESS = self.fileNameESS)


        [n_edge, edge_list, edge_set, edge_dict,
        n_line, line_list, line_set, line_dict,
        n_switch, switch_list, switch_set, switch_dict,
        n_regulator, regulator_list, regulator_set, regulator_dict,
        n_node, node_list, node_set, node_dict,
        n_gen, gen_list, gen_set, gen_dict,
        n_loadcap, loadcap_list, loadcap_set, loadcap_dict,
        n_load, load_list, load_set, load_dict,
        n_cap, cap_list, cap_set, cap_dict,
        n_ess, ess_list, ess_set, ess_dict] = data_import()

        self.textBrowser.clear()
        cursor = self.textBrowser.textCursor()
        cursor.insertHtml('''<p><span style="color: crimson ;"> <h3>Number of nodes: {} </h3>  <br /> </span>'''.format(n_node))
        cursor.insertHtml('''<p><span style="color: green;"><h3>Number of lines: {} </h3>  <br /> </span>'''.format(n_line))
        cursor.insertHtml('''<p><span style="color: blue ;">{} <br /> </span>'''.format(self.generate_table_switch(switch_dict)))
        cursor.insertHtml('''<p><span style="color: lightsalmon ;">{} <br /> </span>'''.format(self.generate_table_cap(cap_dict)))
        cursor.insertHtml('''<p><span style="color: lightseagreen;">{}  <br /> </span>'''.format(self.generate_table_gen(gen_dict)))
        cursor.insertHtml('''<p><span style="color: indigo ;">{} <br /> </span>'''.format(self.generate_table_ess(ess_dict)))




    def default_click(self):
        #self.delete_pickle()

        cwd = os.getcwd()
        self.fileNameNode = cwd + r'/xlsx_data/node data_py.xlsx' 
        self.fileNameLine = cwd + r'/xlsx_data/line data_py.xlsx' 
        self.fileNameSwitch = cwd + r'/xlsx_data/switch data_py.xlsx'   
        self.fileNameRegulator = cwd + r'/xlsx_data/Regulator Data_py.xlsx' 
        self.fileNameLoad = cwd + r'/xlsx_data/spot loads data_py.xlsx'      
        self.fileNameCapacitor = cwd + r'/xlsx_data/cap data_py.xlsx'                    
        self.fileNameDG = cwd + r'/xlsx_data/DG_py.xlsx'        
        self.fileNameESS = cwd + r'/xlsx_data/ESS_py.xlsx'   

        self.lineEdit_node.setText(self.fileNameNode)
        self.lineEdit_line.setText(self.fileNameLine)
        self.lineEdit_switch.setText(self.fileNameSwitch)
        self.lineEdit_regulator.setText(self.fileNameRegulator)
        self.lineEdit_load.setText(self.fileNameLoad)
        self.lineEdit_capacitor.setText(self.fileNameCapacitor)
        self.lineEdit_dg.setText(self.fileNameDG)
        self.lineEdit_ess.setText(self.fileNameESS)



    def summary_click_dss(self):
        #self.delete_pickle()

        AllBusNames , AllLoadNames, AllLineNames, AllTransNames,  AllCapacitorNames, AllTransNames, AllSubNames, Circuit = import_dss_data(self.data_pickle)


        Name = Circuit['Name']
        VoltageBases  = Circuit['VoltageBases']
        NumBuses = Circuit['NumBuses']
        NumNodes = Circuit['NumNodes']
        NumLines = Circuit['NumLines']
        NumLoads = Circuit['NumLoads']
        NumTrans = Circuit['NumTrans']
        NumRegs  = Circuit['NumRegs']
        NumCaps  = Circuit['NumCaps']

        self.textBrowser.clear()
        cursor = self.textBrowser.textCursor()
        cursor.insertHtml('''<p><span style="color: crimson ;"> <h3>Name: {} </h3>  <br /> </span>'''.format(Name))
        cursor.insertHtml('''<p><span style="color: green;"><h3>Voltage Bases: {} </h3>  <br /> </span>'''.format(VoltageBases))
        cursor.insertHtml('''<p><span style="color: blue ;"> <h3>Number of nodes: {} </h3>  <br /> </span>'''.format(NumBuses))
        cursor.insertHtml('''<p><span style="color: lightsalmon;"><h3>Number of nodes: {} </h3>  <br /> </span>'''.format(NumNodes))
        cursor.insertHtml('''<p><span style="color: lightseagreen ;"> <h3>Number of lines: {} </h3>  <br /> </span>'''.format(NumLines))
        cursor.insertHtml('''<p><span style="color: indigo;"><h3>Number of loads: {} </h3>  <br /> </span>'''.format(NumLoads))
        cursor.insertHtml('''<p><span style="color: skylue;"> <h3>Number of transformers: {} </h3>  <br /> </span>'''.format(NumTrans))
        cursor.insertHtml('''<p><span style="color: crimson;"><h3>Number of regulators: {} </h3>  <br /> </span>'''.format(NumRegs))
        cursor.insertHtml('''<p><span style="color: green;"><h3>Number of capacitors: {} </h3>  <br /> </span>'''.format(NumCaps))


    def openFileNameDialogDSS(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.fileNameDSS, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Dat Files (*.dat)", options=options)
        if self.fileNameDSS:
            print(self.fileNameDSS)

        self.lineEdit_dss.setText(self.fileNameDSS)
        self.data_pickle =  self.fileNameDSS


        '''
        data_read(fileNameNode = self.fileNameNode, fileNameLine = self.fileNameLine, 
                fileNameSwitch = self.fileNameSwitch, fileNameRegulator = self.fileNameRegulator, 
                fileNameLoad = self.fileNameLoad, fileNameCapacitor = self.fileNameCapacitor, 
                fileNameDG = self.fileNameDG, fileNameESS = self.fileNameESS)

        [n_edge, edge_list, edge_set, edge_dict,
        n_line, line_list, line_set, line_dict,
        n_switch, switch_list, switch_set, switch_dict,
        n_regulator, regulator_list, regulator_set, regulator_dict,
        n_node, node_list, node_set, node_dict,
        n_gen, gen_list, gen_set, gen_dict,
        n_loadcap, loadcap_list, loadcap_set, loadcap_dict,
        n_load, load_list, load_set, load_dict,
        n_cap, cap_list, cap_set, cap_dict,
        n_ess, ess_list, ess_set, ess_dict] = data_import()
        '''

        #self.textBrowser.clear()
        #cursor = self.textBrowser.textCursor()
        #cursor.insertHtml('''<p><span style="color: crimson ;"> <h3>Number of nodes: {} </h3>  <br /> </span>'''.format(n_node))
        #cursor.insertHtml('''<p><span style="color: green;"><h3>Number of lines: {} </h3>  <br /> </span>'''.format(n_line))
        #cursor.insertHtml('''<p><span style="color: blue ;">{} <br /> </span>'''.format(self.generate_table_switch(switch_dict)))
        #cursor.insertHtml('''<p><span style="color: lightsalmon ;">{} <br /> </span>'''.format(self.generate_table_cap(cap_dict)))
        #cursor.insertHtml('''<p><span style="color: lightseagreen;">{}  <br /> </span>'''.format(self.generate_table_gen(gen_dict)))
        #cursor.insertHtml('''<p><span style="color: indigo ;">{} <br /> </span>'''.format(self.generate_table_ess(ess_dict)))



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


    def generate_table_gen(self, dic):
        table = """
        <h3>GEN configuration</h3>
        <table style="width:100%">
        <tr>
            <th>Location</th>
            <th>Capacity</th> 
        </tr>
        {% for item in items %}
            <tr>
                <td>{{ item.Location }}</td>
                <td>{{ item.Capacity }}</td>
            </tr>
        {% endfor %}
        </table>
        """

        template = Template(table)
        items = []
        for key in dic.keys():
            item = dict(Location=dic[key]['Name'],  Capacity=dic[key]['Pcap'])
            items.append(item)
    

        t = template.render(items=items)
        t = t.replace("\n","")
        return t

                            
    def generate_table_switch(self, dic):
        table = """
        <h3>Switch configuration</h3>
        <table style="width:100%">
        <tr>
            <th>Location </th>
            <th>Status </th> 
            <th>Outage </th> 
        </tr>
        {% for item in items %}
            <tr>
                <td>{{ item.Location }}</td>
                <td>{{ item.Status }}</td>
                <td>{{ item.Outage }}</td>
            </tr>
        {% endfor %}
        </table>
        """

        template = Template(table)
        items = []
        for key in dic.keys():
            item = dict(Location=(dic[key]['Node_A'], dic[key]['Node_B']),  Status=dic[key]['Status'], Outage=dic[key]['Outage'])
            items.append(item)
    

        t = template.render(items=items)
        t = t.replace("\n","")
        return t


    def generate_table_ess(self, dic):
        table = """
        <h3>ESS configuration</h3>
        <table style="width:100%">
        <tr>
            <th>Location </th>
            <th>Capacity </th> 
            <th>min/max SOC </th>
            <th>min/max ch </th>
            <th>min/max disch </th>
            <th>ch/disch effic </th>
        </tr>
        {% for item in items %}
            <tr>
                <td>{{ item.Location }}</td>
                <td>{{ item.Capacity  }}</td>
                <td>{{ item.min_max_SOC }}</td>
                <td>{{ item.min_max_ch }}</td>
                <td>{{ item.min_max_disch }}</td>
                <td>{{ item.ch_disch_effi }}</td>
            </tr>
        {% endfor %}
        </table>
        """

        template = Template(table)
        items = []
        for key in dic.keys():
            item = dict(Location=dic[key]['Name'],  
                Capacity=dic[key]['Capacity'], 
                min_max_SOC = (dic[key]['SOC_min'], dic[key]['SOC_max']),
                min_max_ch = (dic[key]['P_ch_min'],dic[key]['P_ch_max']),
                min_max_disch = (dic[key]['P_disch_min'], dic[key]['P_disch_max']),
                ch_disch_effi = (dic[key]['effi_ch'], dic[key]['effi_disch']))
            items.append(item)
    

        t = template.render(items=items)
        t = t.replace("\n","")
        return t



    def generate_table_cap(self, dic):
        table = """
        <h3>Capacitor bank configuration</h3>
        <table style="width:100%">
        <tr>
            <th>Location</th>
        </tr>
        {% for item in items %}
            <tr>
                <td>{{ item.Location }}</td>
            </tr>
        {% endfor %}
        </table>
        """

        template = Template(table)
        items = []
        for key in dic.keys():
            item = dict(Location=dic[key]['Node'])
            items.append(item)
    

        t = template.render(items=items)
        t = t.replace("\n","")
        return t



    def delete_pickle(self):
        #cwd = os.getcwd() 
        #myfile = cwd + r"/data.dat"
        #myfile = cwd + "/output/data.dat"
        current_path = os.getcwd()
        data_folder = os.path.join(current_path,"output")
        myfile = os.path.join(data_folder, "data.dat")

        ## If file exists, delete it ##
        if os.path.isfile(myfile):
            os.remove(myfile)
        else:    ## Show an error ##
            print("Error: %s file not found" % myfile)



    def read_xls(self, filePath):
        xlsx = pd.ExcelFile(filePath)
        df = xlsx.parse("Sheet1")
        return df


    def gen_dict(self, filePath):
        df = self.read_xls(filePath)
        dic = {}
        header_list = list(df.columns.values)
    
        for i in header_list:
            dic[i] = df[i].values.tolist()      
        return dic



    def write_pickle(self, filePath):
        dic = self.gen_dict(filePath)
        print('Saving data to pickle file...')
        try:
            with open(data_pickle, 'wb') as pfile:
                pickle.dump(dic, pfile, pickle.HIGHEST_PROTOCOL)
        
        except Exception as e:
            print('Unable to save data to', data_pickle, ':', e)
            raise
        print('Data cached in pickle file.')



    def read_pickle(self, data_pickle):
        with open(data_pickle, 'rb') as f:
            datadict = pickle.load(f)
            print(datadict)



class popupWindow(QtWidgets.QWidget):
    resized = QtCore.pyqtSignal()

    def __init__(self, topologyFile, topologyFilePath):
        #super().__init__()
        super(popupWindow, self).__init__()
        self.title = 'Topology of Distribution Networks for ' + str(topologyFile)
        self.left = 10
        self.top = 10
        self.width = 800 #640
        self.height = 800 #480
        #self.topologyFile = topologyFile
        self.topologyFilePath = topologyFilePath

        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label = QtWidgets.QLabel(self) 

        self.label.resize(self.width, self.height)
        self.pixmap = QtGui.QPixmap(self.topologyFilePath)
        self.pixmap = self.pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation);
        self.label.setPixmap(self.pixmap)
        ##label.setMinimumSize(1, 1)
        ##label.installEventFiler(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setScaledContents(True)
        #self.label.eventFilter(self)

        #self.resizeEvent()
        self.resized.connect(self.someFunction)


        #self.resize(self.pixmap.width(),self.pixmap.height())
 
        #self.show()
 


    def resizeEvent(self, event):
        self.resized.emit()
        #return super(popupWindow, self).resizeEvent(event)

    def someFunction(self):
        #print("someFunction")
        #width = self.label.frameGeometry().width()
        #height = self.label.frameGeometry().height()
        #print(width, height)
        pixmap = QtGui.QPixmap(self.topologyFilePath)
        pixmap = pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)
        self.label.resize(self.width, self.height)
        #print(popupWindow.size())
        #label = QtWidgets.QLabel(self) 
        #label.resize(self.width, self.height)
        #pixmap = QtGui.QPixmap(self.topologyFilePath)
        #pixmap = pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation);
        #self.label.setPixmap(pixmap)
        #label.setMinimumSize(1, 1)
        #label.installEventFiler(self)
        #label.setAlignment(QtCore.Qt.AlignCenter)
        #label.setScaledContents(True)
    '''

        

    def eventFilter(self, source, event):
        if (source is self.label and event.type() == QtCore.QEvent.Resize):

            # re-scale the pixmap when the label resizes
            self.label.setPixmap(self.pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        return super(popupWindow, self).eventFilter(source, event)
    '''
    '''
    def resizeEvent(self, event):
        pixmap = QtGui.QPixmap(self.topologyFilePath)
        pixmap = pixmap.scaled(self.width(), self.height())
        label.setPixmap(self.pixmap)
        label.resize(self.width(), self.height())
    '''
      

    '''
    def installEventFiler(self, source, event):
        if (source is self.ui.label and event.type() == QtCore.QEvent.Resize):
            # re-scale the pixmap when the label resizes
            self.label.setPixmap(self.pixmap.scaled( self.label.size(), QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation))

        return super(popupWindow, self).eventFilter(source, event)
        
        #self.resize(pixmap.width(),pixmap.height())
    '''


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui_Window_1 = GUI_Window_1()
    gui_Window_1.setStyle('Macintosh')
    palette = QtGui.QPalette()
    #palette.setColor(QPalette.ButtonText, Qt.red)
    gui_Window_1.setPalette(palette)
    gui_Window_1.show()
    sys.exit(app.exec_())
