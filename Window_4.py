
# Build upon the pyqt skeleton available at: https://github.com/shantnu/PyQt_first/blob/master/pyqt_skeleton.py
# [09/15/2017] Adapt PyQt4 code to PyQt5. Have the error "'module' object has no attribute 'QtGui.QMainWindow'"
# [09/15/2017] Solution: Change QtGui.QMainWindow to QtWidgets.QMainWindow.
# [09/15/2017] More Differences between Qt4 and Qt5:
# [09/15/2017] http://pyqt.sourceforge.net/Docs/PyQt5/pyqt4_differences.html
# [09/15/2017] https://www.zhihu.com/question/50630092
# [09/18/2017] Designed GUI interface, receive data from user input
# [09/19/2017] Receive data and sovle DSR problem
# [04/20/2018] Change the layout of the GUI. Integrate all the sub-windows into the main window.


import sys
import os
from glob import glob
from time import strftime, localtime
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import numpy as np
import imageio
import restoration_module as solve_module
import visualization_module as visualize_module
import pathlib

current_path = os.getcwd()
data_folder = os.path.join(current_path,"ui_forms")
qtWindow_4File = os.path.join(data_folder, "Window_4.ui")

#qtWindow_4File = r"./ui_forms/Window_4.ui"
Ui_Window_4, QtBaseClass = uic.loadUiType(qtWindow_4File)


'''
class Table(QTableWidget):
  def sizeHint(self):
    horizontal = self.horizontalHeader()
    vertical = self.verticalHeader()
    frame = self.frameWidth() * 2
    return QSize(horizontal.length() + vertical.width() + frame,
                 vertical.length() + horizontal.height() + frame)
'''

class DSR(QtWidgets.QMainWindow, Ui_Window_4):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_Window_4.__init__(self)
        self.setupUi(self)

        ######################
        # TODO: disable some buttons?
        ######################
        message = strftime("[%H:%M:%S]: Initialization Completed.", localtime())
        self.log_box.setText(message)
        ######################
        self.solve_btn.pressed.connect(self.readinput)
        ######################
        self.solve_btn.pressed.connect(self.solve_sr)
        ######################
        #self.visualization_btn.pressed.connect(self.diagram_prepare)
        ######################

        #self.visualization_btn.pressed.connect(self.plot_diagram)
        self.animation_btn.pressed.connect(self.plot_animated_diagram)


        self.animation_btn_2.pressed.connect(self.plot_animation)
        #self.plot_diagram_btn.pressed.connect(self.plot_diagram)
        #self.plot_load_profile_btn.pressed.connect(self.plot_load)


        #sys.stdout = port(self.log_box)


        #https://github.com/pyqt/examples/blob/master/widgets/sliders.py#L47
        #https://www.programcreek.com/python/example/108074/PyQt5.QtWidgets.QScrollBar
        #https://stackoverflow.com/questions/6194659/configuring-a-custom-scrollbar-in-pyqt
        #https://www.programcreek.com/python/example/101693/PyQt5.QtCore.Qt.Horizontal

        self.minimumSpinBox.setValue(0)
        self.maximumSpinBox.setValue(10)
        self.valueSpinBox.setValue(5)

        self.createControls()

        self.horizontalScrollBar.valueChanged.connect(self.plot_sequence_diagram)

        self.movie = QtGui.QMovie(self)

        '''
        top = Table(3, 5, self)
        self.setCentralWidget(top)
        '''

    def createControls(self):
  
        #valueChanged = QtCore.pyqtSignal(int)
        self.horizontalScrollBar.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        self.horizontalSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.horizontalSlider.setTickInterval(10)
        self.horizontalSlider.setSingleStep(1)


        self.horizontalSlider.valueChanged.connect(self.horizontalScrollBar.setValue)
        self.horizontalScrollBar.valueChanged.connect(self.horizontalSlider.setValue)

        self.horizontalSlider.valueChanged.connect(self.valueSpinBox.setValue)
        self.valueSpinBox.valueChanged.connect(self.horizontalScrollBar.setValue)


        self.minimumSpinBox.setRange(0, 11) #-100, 100
        self.minimumSpinBox.setSingleStep(1)

        self.maximumSpinBox.setRange(0, 11) #-100, 100
        self.maximumSpinBox.setSingleStep(1)

        self.valueSpinBox.setRange(0, 11) #-100, 100
        self.valueSpinBox.setSingleStep(1)

        self.minimumSpinBox.valueChanged.connect(self.horizontalSlider.setMinimum)
        self.maximumSpinBox.valueChanged.connect(self.horizontalSlider.setMaximum)


        self.minimumSpinBox.valueChanged.connect(self.horizontalScrollBar.setMinimum)
        self.maximumSpinBox.valueChanged.connect(self.horizontalScrollBar.setMaximum)


    def readinput(self):
        # System Information User Input
        pf_en = self.power_flow.isChecked()
        volt_en = self.voltage_limit.isChecked()
        line_cap_en = self.line_capacity.isChecked()
        gen_cap_en = self.gen_capacity.isChecked()
        gen_reserve_en = self.gen_reserve.isChecked()
        gen_ramp_en = self.gen_ramp.isChecked()
        gen_stepload_en = self.gen_stepload.isChecked()
        gen_unbalance_en = self.gen_unbalance.isChecked()
        non_shed_en = self.non_shed.isChecked()
        topology_en = self.topology.isChecked()

        model_config = {"st_pf_balance": pf_en,
                        "st_pf_voltage": pf_en,
                        "st_voltage_limit": volt_en,
                        "st_line_capacity": line_cap_en,
                        "st_gen_capacity": gen_cap_en,
                        "st_gen_reserve": gen_reserve_en,
                        "st_gen_ramp": gen_ramp_en,
                        "st_gen_stepload": gen_stepload_en,
                        "st_gen_unbalance": gen_unbalance_en,
                        "st_non_shed": non_shed_en,
                        "st_connectivity": topology_en,
                        "st_sequence": topology_en,
                        "st_topology": topology_en}


        rh_start_time = "13:00"
        rh_horizon = self.horizon_step.value()  # total steps in each iteration
        rh_control = self.control_step.value()  # within each iteration, how many steps to carry out
        rh_set_step = self.total_step.value()  # steps set by the user
        rh_step_length = self.length_step.value()  # in minute

        rh_iteration = int(np.ceil((rh_set_step-rh_horizon)/(rh_control-1) + 1))
        rh_total_step = (rh_control-1)*(rh_iteration-1) + rh_horizon  # total steps used by the algorithm
        rh = {'rh_start_time': rh_start_time,
              'rh_horizon': rh_horizon,
              'rh_control': rh_control,
              'rh_set_step': rh_set_step,
              'rh_step_length': rh_step_length,
              'rh_iteration': rh_iteration,
              'rh_total_step': rh_total_step,
              'rh_model_config': model_config}


        # Problem Formulation User Input
        sr_clpu_enable = self.clpu_enable.isChecked()  # enable Cold Load Pick Up load model
        sr_es_enable = self.ess_enable.isChecked()  # enable considering ESS model
        sr_rg_enable = self.vg_enable.isChecked()  # enable considering voltage regulator
        sr_cap_enable = self.cb_enable.isChecked()  # enable considering capacitor banks
        sr_Vbase = self.Vbase.value()*1000  # L-L voltage
        sr_Sbase = self.Sbase.value()*1000  # kVA

        sr_re_enable = False  # enable considering renewable energies
        sr_n_polygon = 4  # number of polygen to approximate x^2 + y^2 <= C
        sr_Vsrc = 1.05  # expected voltage in per unit of the black-start DG
        sr_M = 10000  # value used in the big-M method.
        sr_reserve_margin = 0.15  # capacity margin for each DG
        sr = {'sr_clpu_enable': sr_clpu_enable,
              'sr_re_enable': sr_re_enable,
              'sr_es_enable': sr_es_enable,
              'sr_rg_enable': sr_rg_enable,
              'sr_cap_enable': sr_cap_enable,
              'sr_Vbase': sr_Vbase,
              'sr_Sbase': sr_Sbase,
              'sr_n_polygon': sr_n_polygon,
              'sr_Vsrc': sr_Vsrc,
              'sr_M': sr_M,
              'sr_reserve_margin': sr_reserve_margin}

        self.model_config = rh
        self.DSR_config = sr

        message = strftime("[%H:%M:%S]: User Input Data Received.", localtime())
        self.log_box.append(message)

    def solve_sr(self):
        try:
            solve_module.solve(self.model_config, self.DSR_config)
            message = strftime("[%H:%M:%S]: DSR problem solved.", localtime())
            self.log_box.append(message)
        # TODO: if infeasible solution, then disable some constriants to generate less feasible solutions?
        except:
            message = strftime("[%H:%M:%S]: Failed to solve DSR problem.", localtime())
            self.log_box.append(message)


    def diagram_prepare(self):
        try:
            [n_edge, edge_list, edge_set, edge_dict,
             n_line, line_list, line_set, line_dict,
             n_switch, switch_list, switch_set, switch_dict,
             n_regulator, regulator_list, regulator_set, regulator_dict,
             n_node, node_list, node_set, node_dict,
             n_gen, gen_list, gen_set, gen_dict,
             n_loadcap, loadcap_list, loadcap_set, loadcap_dict,
             n_load, load_list, load_set, load_dict,
             n_cap, cap_list, cap_set, cap_dict,
             n_ess, ess_list, ess_set, ess_dict] = solve_module.data_import()

            [gen, P_A_gen, P_B_gen, P_C_gen, Q_A_gen, Q_B_gen, Q_C_gen,
             edge,
             load, P_A_load, P_B_load, P_C_load, Q_A_load, Q_B_load, Q_C_load,
             node,
             ess_ch, ess_disch,
             P_A_ess_ch, P_B_ess_ch, P_C_ess_ch, Q_A_ess_ch, Q_B_ess_ch, Q_C_ess_ch,
             P_A_ess_disch, P_B_ess_disch, P_C_ess_disch, Q_A_ess_disch, Q_B_ess_disch, Q_C_ess_disch,
             SOC_A, SOC_B, SOC_C,
             rh_start_time, rh_horizon, rh_control, rh_set_step, rh_step_length, rh_iteration, rh_total_step,
             sr_clpu_enable, sr_re_enable, sr_es_enable, sr_rg_enable, sr_Vbase, sr_Sbase, sr_cap_enable,
             sr_n_polygon, sr_Vsrc, sr_M, sr_reserve_margin] = visualize_module.solution_import()

            self.loadcap_dict = loadcap_dict
            self.rh_total_step = rh_total_step
            ############################
            steps = [str(i) for i in range(self.rh_total_step)]
            self.diagram_step_select.addItems(steps)
            self.load_name.addItems(loadcap_set)

            # ############################
            # self.plot_diagram_btn.pressed.connect(self.plot_diagram)
            # ############################
            # self.plot_load_profile_btn.pressed.connect(self.plot_load)
            # ############################
            # self.show()
            message = strftime("[%H:%M:%S]: Diagrams generated.", localtime())
            self.log_box.append(message)
        except:
            message = strftime("[%H:%M:%S]: Failed to generate diagrams.", localtime())
            self.log_box.append(message)




    def plot_diagram(self):
        step_select = int(self.diagram_step_select.currentText())
        visualize_module.plot_sequence(step_select)
        pic = QtGui.QPixmap("diagram_step.png")
        self.diagram_label.setPixmap(pic)
        self.diagram_label.setScaledContents(True)



    def plot_animated_diagram(self):
      
      #images = []

      for i in range(11):
        visualize_module.plot_sequence(i)
        new_name = "output/diagram_step" + "_" + str(i) + ".png"
        os.rename("output/diagram_step.png", new_name)

        #images.append(new_name)

  
      #self.generate_animated_image(images, "output/diagram_step.gif")
      


      #width = self.diagram_label.frameGeometry().width()
      #height = self.diagram_label.frameGeometry().height()

      #https://www.programcreek.com/python/example/106692/PyQt5.QtGui.QMovie
      #movie = QtGui.QMovie("output/diagram_step.gif")
      #movie.setScaledSize(QtCore.QSize(width, height))
      #movie.frameChanged.connect(lambda: self.diagram_label.setPixmap(movie.currentPixmap()))
      #self.diagram_label.setMovie(movie)
      #movie.start()



      #if self.horizontalScrollBar.valueChanged:
      #  movie.stop()

      #self.novideoMovie = QtGui.QMovie("diagram_step.gif", b'GIF', self)
      #width = self.diagram_label.frameGeometry().width()
      #height = self.diagram_label.frameGeometry().height()

      #self.novideoMovie.setScaledSize(QtCore.QSize(width, height))
      #self.novideoMovie.frameChanged.connect(lambda: self.diagram_label.setPixmap(self.novideoMovie.currentPixmap()))
      #self.novideoMovie.start()

      #self.movie = QtGui.QMovie(self)
      #self.movie.setFileName("diagram_step.gif")
      #self.movie.setCacheMode(QtGui.QMovie.CacheAll)
      #self.movie.start()
      #self.diagram_label.setAlignment(QtCore.Qt.AlignCenter)
      #self.diagram_label.setMovie(self.movie)
      #self.diagram_label.hide() 



    def plot_animation(self):

      images = glob("./output/diagram_step_*.png")
  
      self.generate_animated_image(images, "output/diagram_step.gif")
      
      width = self.diagram_label.frameGeometry().width()
      height = self.diagram_label.frameGeometry().height()

      #https://www.programcreek.com/python/example/106692/PyQt5.QtGui.QMovie
      #self.movie = QtGui.QMovie("./output/diagram_step.gif")
      #self.movie = QtGui.QMovie(self)
      self.movie.setFileName("./output/diagram_step.gif")
      self.movie.setScaledSize(QtCore.QSize(width, height))
      self.movie.frameChanged.connect(lambda: self.diagram_label.setPixmap(self.movie.currentPixmap()))
      self.diagram_label.setMovie(self.movie)
      self.movie.start()




    #https://stackoverflow.com/questions/38433425/custom-frame-duration-for-animated-gif-in-python-imageio
    def generate_animated_image(self, inputFileNames, outputFileName):
      import imageio
      frames = []
      for filename in inputFileNames:
        frames.append(imageio.imread(filename))
      imageio.mimsave(outputFileName , frames, format='GIF', duration=1.0)
      


    def plot_sequence_diagram(self, value):
      imageFile = "output/diagram_step_"  + str(value) + ".png"
      pic = QtGui.QPixmap(imageFile)
      self.diagram_label.setPixmap(pic)
      self.diagram_label.setScaledContents(True)
      self.show()

      self.movie.stop()


    def plot_load(self):
      load_name = self.load_name.currentText()
      node_name = self.loadcap_dict[load_name]['Node']
      visualize_module.plot_load(node_name)
      pic = QtGui.QPixmap("output/load_profile.png")
      self.load_label.setPixmap(pic)
      self.load_label.setScaledContents(True)



class port(object):
    def __init__(self,view):
        self.view = view

    def write(self,*args):
        self.view.append(*args)

    def flush(self):
        pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = DSR()
    mainwindow.show()
    sys.exit(app.exec_())