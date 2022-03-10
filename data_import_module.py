#!/user/bin/env python3
# -*- coding: utf-8 -*-
from xlrd import open_workbook
import numpy as np
import pickle
import os

##############################################################################################
# Model the set of line as a class, with all the attributes as its properties and methods
class File(object):
    def __init__(self, file_type, file_location):
        self.file_type = file_type
        self.file_location = file_location

def file_selection():
    # TODO Implement in the GUI to allow customers to select the file and sheet number to be imported
    # return component_type file_type and file_location
    pass

def file_location_correction(file_location):
    #  TODO Correct file formats when different formats are supported
    #  [08/03/2017] xlrd only recognizes 'xls' formats. Replace 'xlsx' with 'xls'
    if file_location.lower().endswith('.xls'):
        file_type = 'xls'
    elif file_location.lower().endswith('.xlsx'):
        file_type = 'xls'
    else:
        file_type = 'xls'
    file_location_corrected = file_location.replace('.xlsx', '.xls')
    return [file_type, file_location_corrected]

def sheets_selection(wb):
    # [08/03/2017] Assume all the data is on the single sheet
    if wb.nsheets > 1:
        sheets_index = 0
    else:
        sheets_index = 0
    sheets = wb.sheets()[sheets_index]
    return sheets

##############################################################################################
# Define the class for line
class InfoLine(object):
    # Default values:  Length_ft: -1 means there is an error if info is not updated
    # Config: '-1' as a string, also means an error
    # Switch: 2 is default setting, means this line is not switchable
    def __init__(self, Name='NoName', Node_A='-1', Node_B='-1', Length_ft=-1, Config='-1',
                 Impedance = np.zeros((3,3)), Ampacity = 0.0,
                 Phases=np.array([1, 1, 1]), Control=False, Status=False, Outage=False):
        self.Name = Name
        self.Node_A = Node_A
        self.Node_B = Node_B
        self.Length_ft = Length_ft
        self.Config = Config
        self.Impedance = Impedance
        self.Ampacity = Ampacity
        self.Phases = Phases
        self.Control = Control  # if a line can be controlled remotely
        self.Status = Status    # if a line is On (closed) or Off (opened)
        self.Outage = Outage    # if a line is damaged during the outage

# Define the class for switch
class InfoSwitch(object):
    # Default values:
    # Control: 'No' means this switch is damaged and unoperable
    # Status: 'closed' is normal status of this switch
    def __init__(self, Name='NoName', Node_A='-1', Node_B='-1', Impedance = np.zeros((3,3)), Ampacity = 0,
                 Phases=np.array([1, 1, 1]), Control=True, Status=True, Outage=False):
        self.Name = Name
        self.Node_A = Node_A
        self.Node_B = Node_B
        self.Impedance = Impedance
        self.Ampacity = Ampacity
        self.Phases = Phases
        self.Control = Control
        self.Status = Status
        self.Outage = Outage

# Define the class for regulator
class InfoRegulator(object):
    def __init__(self, Name='NoName', Line='N-N', Node='None', Node_A='None', Node_B='None', Impedance = np.zeros((3,3)), Ampacity = 0,
                 Phases=np.array([1, 1, 1]),Control=False, Status=False, Outage=False):
        self.Name = Name
        self.Line = Line
        self.Node = Node
        self.Node_A = Node_A
        self.Node_B = Node_B
        self.Impedance = Impedance
        self.Ampacity = Ampacity
        self.Phases = Phases
        self.Control = Control
        self.Status = Status
        self.Outage = Outage

# Define the class for loads ( and loadcap)
class InfoLoad(object):
    def __init__(self, Name, Node, ZIP, Conn, P=np.zeros((3,)), Q=np.zeros((3,)), Phases=np.array([1, 1, 1]),
                 Control=False, Status=False, Outage=False, CLPU_factor = [2.5, 1.0, 2.0, 0.3]):
        self.Name = Name
        self.Node = Node
        self.ZIP = ZIP
        self.Conn = Conn
        self.P = P
        self.Q = Q
        self.Phases = Phases
        self.Control = Control
        self.Status = Status
        self.Outage = Outage
        self.CLPU_factor = CLPU_factor  # CLPU_factor = [SU, SD, DIV, decay]  DIV

    # Generate CLPU load profiles (CLPU factors are pre-determined, one can modify this later)
    def clpu_generation(self, step_length):
        clpu_window_size = 10000 # in seconds
        sample_step_length = int(step_length * 60) # convert to seconds
        if "load" in self.Name:
            overshot = self.CLPU_factor[0]
            diverse = self.CLPU_factor[1]
            undiv = self.CLPU_factor[2] * 60 # convert to seconds
            decay = self.CLPU_factor[3] / 60 # scale to second-based decay curve
        elif "cap" in self.Name:
            overshot = 1
            diverse = 1
            undiv = clpu_window_size
            decay = 0
            self.CLPU_factor = [1, 1, 0, 0]
        decay_sample = (overshot - diverse) * np.exp(-1 * decay * np.arange(0, clpu_window_size))
        undiv_length = np.ravel(np.ones([1, int(undiv)]))
        decay_profile = np.concatenate((overshot * undiv_length, 1 + decay_sample))
        decay_difference = []
        for i in range(0, clpu_window_size+sample_step_length, sample_step_length):
            decay_difference.append(decay_profile[i+sample_step_length] - decay_profile[i])
        self.CLPU_profile = decay_difference

# Define the class for capacitor
class InfoCap(object):
    def __init__(self, Name, Node, P, Q, Phases=np.array([1, 1, 1]), Control=False, Status=False, Outage=False):
        self.Name = Name
        self.Node = Node
        self.P = P
        self.Q = Q
        self.Phases = Phases
        self.Control = Control
        self.Status = Status
        self.Outage = Outage

# Define the class for node
class InfoNode(object):
    def __init__(self, Name, Node, Number, Phases=np.array([1, 1, 1]), GIS=[0.0,0.0], Control=False, Status=False, Outage=False):
        self.Name = Name
        self.Node = Node
        self.Number = Number  # numbering nodes, index of the vector
        self.Phases = Phases
        self.GIS = GIS
        self.Control = Control
        self.Status = Status
        self.Outage = Outage

# Define the class for generator
class InfoGen(object):
    def __init__(self, Name, Node, Type, Capacity, Pmax, Pmin, Qmax, Qmin, Rmax, FRR, CUI, Phases=np.array([1, 1, 1]), Control=False, Status=False, Outage=False):
        self.Name = Name
        self.Node = Node
        self.Type = Type
        self.Pcap = Capacity
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Qmax = Qmax
        self.Qmin = Qmin
        self.Rmax = Rmax
        self.FRR = FRR
        self.CUI = CUI
        self.Phases = Phases
        self.Control = Control
        self.Status = Status
        self.Outage = Outage

# Define the class for ESS
class InfoESS(object):
    def __init__(self, Name, Node, Phases, Capacity, SOC_init, SOC_min, SOC_max, P_rated, P_ch_min, P_ch_max,
                 P_disch_min, P_disch_max, effi_ch, effi_disch, idle_P_loss, idle_Q_loss, Control, Status, Outage):
        self.Name = Name
        self.Node = Node
        self.Phases = Phases
        self.Capacity = Capacity
        self.SOC_init = SOC_init # NOTE: this is the init for each phase
        self.SOC_min  = SOC_min
        self.SOC_max  = SOC_max
        self.P_rated  = P_rated
        self.P_ch_min = P_ch_min
        self.P_ch_max = P_ch_max
        self.P_disch_min = P_disch_min
        self.P_disch_max = P_disch_max
        self.effi_ch = effi_ch
        self.effi_disch = effi_disch
        self.idle_P_loss = idle_P_loss
        self.idle_Q_loss = idle_Q_loss
        self.Control = Control
        self.Status = Status
        self.Outage = Outage

# Define the linecode
class linecode_library(object):
    def __init__(self):
        self.line_matrix = {"1": [[1,1,1],   # phasing
                                  np.array([[0.4576+1.0780j, 0, 0], # impedance per mile
                                           [0, 0.4576+1.0780j, 0],
                                           [0, 0, 0.4576+1.0780j]]),
                                  530],  # current capacity, or ampacity
                            "2": [[1,1,1],
                                  np.array([[0.4666+1.0482j, 0.1580+0.4236j, 0.1560+0.5017j],
                                           [0.1580+0.4236j, 0.4615+1.0651j, 0.1535+0.3849j],
                                           [0.1560+0.5017j, 0.1535+0.3849j, 0.4576+1.0780j]]),
                                  530],
                            "3": [[1,1,1],
                                  np.array([[0.4615+1.0651j, 0.1535+0.3849j, 0.1580+0.4236j],
                                           [0.1535+0.3849j, 0.4576+1.0780j, 0.1560+0.5017j],
                                           [0.1580+0.4236j, 0.1560+0.5017j, 0.4666+1.0482j]]),
                                  530],
                            "4": [[1,1,1],
                                  np.array([[0.4615+1.0651j, 0.1580+0.4236j, 0.1535+0.3849j],
                                           [0.1580+0.4236j, 0.4666+1.0482j, 0.1560+0.5017j],
                                           [0.1535+0.3849j, 0.1560+0.5017j, 0.4576+1.0780j]]),
                                  530],
                            "5": [[1,1,1],
                                  np.array([[0.4666+1.0482j, 0.1560+0.5017j, 0.1580+0.4236j],
                                           [0.1560+0.5017j, 0.4576+1.0780j, 0.1535+0.3849j],
                                           [0.1580+0.4236j, 0.1535+0.3849j, 0.4615+1.0651j]]),
                                  530],
                            "6": [[1,1,1],
                                  np.array([[0.4576+1.0780j, 0.1535+0.3849j, 0.1560+0.5017j],
                                           [0.1535+0.3849j, 0.4615+1.0651j, 0.1580+0.4236j],
                                           [0.1560+0.5017j, 0.1580+0.4236j, 0.4666+1.0482j]]),
                                  530],
                            "7": [[1,0,1],
                                  np.array([[0.4576+1.0780j, 0.0000+0.0000j, 0.1535+0.3849j],
                                           [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                                           [0.1535+0.3849j, 0.0000+0.0000j, 0.4615+1.0651j]]),
                                  530],
                            "8": [[1,1,0],
                                  np.array([[0.4576+1.0780j, 0.1535+0.3849j, 0.0000+0.0000j],
                                           [0.1535+0.3849j, 0.4615+1.0651j, 0.0000+0.0000j],
                                           [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]]),
                                  530],
                            "9": [[1,0,0],
                                  np.array([[1.3292+1.3475j, 0.0000+0.0000j, 0.0000+0.0000j],
                                           [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                                           [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]]),
                                  230],
                            "10":[[0,1,0],
                                  np.array([[0.0000+0.0000j,0.0000+0.0000j, 0.0000+0.0000j],
                                           [0.0000+0.0000j, 1.3292+1.3475j, 0.0000+0.0000j],
                                           [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]]),
                                  230],
                            "11":[[0,0,1],
                                  np.array([[0.0000+0.0000j,0.0000+0.000j,  0.0000+0.0000j],
                                           [0.0000+0.0000j, 0.0000+0.000j,  0.0000+0.0000j],
                                           [0.0000+0.0000j, 0.0000+0.0000j, 1.3292+1.3475j]]),
                                  230],
                            "12":[[1,1,1],
                                  np.array([[1.5209+0.7521j,0.5198+0.2775j, 0.4924+0.2157j],
                                           [0.5198+0.2775j, 1.5329+0.7162j, 0.5198+0.2775j],
                                           [0.4924+0.2157j, 0.5198+0.2775j, 1.5209+0.7521j]]),
                                  155],
                            "XFM-1":[[1,1,1],
                                     np.array([[0.4884 + 1.046j, 0+0j,            0+0j],
                                              [0+0j,             0.4884 + 1.046j, 0+0j],
                                              [0+0j,             0+0j,            0.4884 + 1.046j]]),
                                     400],
                            "regulator":[['?','?','?'],
                                         np.array([[0.4884 + 1.046j, 0 + 0j, 0 + 0j],
                                               [0 + 0j, 0.4884 + 1.046j, 0 + 0j],
                                               [0 + 0j, 0 + 0j, 0.4884 + 1.046j]]),
                                         600],
                            "switch": [[1,1,1],
                                       np.array([[0+0j, 0+0j, 0+0j],
                                                [0+0j, 0+0j, 0+0j],
                                                [0+0j, 0+0j, 0+0j]]),
                                       400]
                            }



##############################################################################################
def getdata_line(sheet_line):
    title_row = title_row_location(sheet_line, 'Line', 'node', 'length')
    node_A_col = col_location(sheet_line, title_row, 'Line', 'node a')
    node_B_col = col_location(sheet_line, title_row, 'Line', 'node b')
    length_col = col_location(sheet_line, title_row, 'Line', 'length')
    config_col = col_location(sheet_line, title_row, 'Line', 'config')
    control_col = col_location(sheet_line, title_row, 'Line', 'control')
    status_col = col_location(sheet_line, title_row, 'Line', 'status')
    outage_col = col_location(sheet_line, title_row, 'Line', 'outage')

    # Model the set of line objects as a dict, all the line names are in a set
    n_line = sheet_line.nrows - title_row - 1
    line=[]
    line_dict = {}     # Not sure which structure is better: dict or list of objects?
    line_set = set([]) # collection of unique elements

    line_matrix = linecode_library().line_matrix

    # Starting from the row below the found title_row, assign values to lines
    for i in range(title_row + 1, sheet_line.nrows):
        [line_name, from_node, to_node] = edge_naming(sheet_line, 'line', i, node_A_col, node_B_col)
        line_length = sheet_line.cell(i, length_col).value
        line_config = num2str(sheet_line.cell(i, config_col).value)
        line_control = bool(sheet_line.cell(i, control_col).value)
        line_status = bool(sheet_line.cell(i, status_col).value)
        line_outage = bool(sheet_line.cell(i, outage_col).value)
        line_phases = line_matrix[line_config][0]
        line_impedance = float(line_length)/5280*line_matrix[line_config][1]
        line_ampacity = line_matrix[line_config][2]

        line_instance = InfoLine(line_name, from_node, to_node, line_length, line_config,
                                 line_impedance, line_ampacity, line_phases, line_control, line_status, line_outage)
        line.append(line_instance)
        line_set.add(line_name)

        line_dict[line_name] = {"Node_A": from_node,
                                "Node_B": to_node,
                                "Length": line_length,
                                "Config": line_config,
                                "Impedance": line_impedance,
                                "Ampacity": line_ampacity,
                                "Phases": line_phases,
                                "Control": line_control,
                                "Status": line_status,
                                "Outage": line_outage}

    return n_line, line, line_set, line_dict

def getdata_switch(sheet_switch):
    title_row =  title_row_location(sheet_switch, 'Switch', 'node', 'status')
    node_A_col =  col_location(sheet_switch, title_row, 'Switch', 'node a')
    node_B_col =  col_location(sheet_switch, title_row, 'Switch', 'node b')
    status_col =  col_location(sheet_switch, title_row, 'Switch', 'status')
    control_col =  col_location(sheet_switch, title_row, 'Switch', 'control')
    outage_col =  col_location(sheet_switch, title_row, 'Switch', 'outage')

    n_switch = sheet_switch.nrows - title_row - 1
    switch=[]
    switch_dict = {}     # Not sure which structure is better: dict or list of objects?
    switch_set = set([]) # collection of unique elements

    line_matrix = linecode_library().line_matrix

    # Starting from the row below the found title_row
    for i in range(title_row + 1, sheet_switch.nrows):
        [switch_name, from_node, to_node] = edge_naming(sheet_switch, 'switch', i, node_A_col, node_B_col)
        switch_phases = line_matrix['switch'][0]
        switch_impedance = line_matrix['switch'][1]
        switch_ampacity = line_matrix['switch'][2]
        switch_control = bool(sheet_switch.cell(i, control_col).value)
        switch_status = bool(sheet_switch.cell(i, status_col).value)
        switch_outage = bool(sheet_switch.cell(i, outage_col).value)

        switch_instance = InfoSwitch(switch_name, from_node, to_node, switch_impedance,
                                     switch_ampacity, switch_phases, switch_control, switch_status, switch_outage)
        switch.append(switch_instance)
        switch_set.add(switch_name)

        switch_dict[switch_name] = {"Node_A": from_node,
                                    "Node_B": to_node,
                                    "Impedance": switch_impedance,
                                    "Ampacity": switch_ampacity,
                                    "Phases": switch_phases,
                                    "Control": switch_control,
                                    "Status": switch_status,
                                    "Outage": switch_outage}

    return n_switch, switch, switch_set, switch_dict

def getdata_regulator(sheet_regulator):
    n_regulator = 0
    regulator_list = []
    regulator_set = set([])
    regulator_dict = {}

    line_matrix = linecode_library().line_matrix

    for row in range(sheet_regulator.nrows):
        data = '  ' + sheet_regulator.cell(row, 0).value.lower()
        if data.find('regulator id') > 0:
            n_regulator += 1
            line = num2str(sheet_regulator.cell(row + 1, 1).value).replace(' ','')
            mid = line.find('-')
            Node_A = line[:mid]
            Node_B = line[(mid + 1):]
            Node = num2str(sheet_regulator.cell(row + 2, 1).value)
            name = 'regulator_' + Node
            Impedance = line_matrix['regulator'][1]
            Ampacity = line_matrix['regulator'][2]
            phases_in_ABC = num2str(sheet_regulator.cell(row + 3, 1).value).replace('-','').replace(' ','')
            phases = ABC_2_111(phases_in_ABC)
            control = bool(sheet_regulator.cell(row + 13, 1).value)
            status = bool(sheet_regulator.cell(row + 14 , 1).value)
            outage = bool(sheet_regulator.cell(row + 15, 1).value)

            regulator_instance = InfoRegulator(name, line, Node, Node_A, Node_B, Impedance, Ampacity, phases, control, status, outage)
            regulator_list.append(regulator_instance)
            regulator_set.add(name)
            regulator_dict[name] = {"Line": line,
                                    "Node_A": Node_A,
                                    "Node_B": Node_B,
                                    "Node": Node,
                                    "Impedance": Impedance,
                                    "Ampacity": Ampacity,
                                    "Phases": phases,
                                    "Control": control,
                                    "Status": status,
                                    "Outage": outage}
    return n_regulator, regulator_list, regulator_set, regulator_dict

def getdata_spotload(sheet_spotload):
    n_load = 0
    load_list = []
    load_set = set([])
    load_dict = {}

    data_row = 4
    for i in range(data_row, sheet_spotload.nrows):
        if sheet_spotload.cell(i, 0).value == 'Total':
            break
        else:
            node = num2str(sheet_spotload.cell(i, 0).value)
            name = 'load_' + node
            model = num2str(sheet_spotload.cell(i, 1).value)
            mid = model.find('-')
            conn = model[0]
            ZIP = model[(mid + 1):]
            P1 = float(sheet_spotload.cell(i, 2).value)
            Q1 = float(sheet_spotload.cell(i, 3).value)
            P2 = float(sheet_spotload.cell(i, 4).value)
            Q2 = float(sheet_spotload.cell(i, 5).value)
            P3 = float(sheet_spotload.cell(i, 6).value)
            Q3 = float(sheet_spotload.cell(i, 7).value)
            P = [P1, P2, P3]
            Q = [Q1, Q2, Q3]
            phases = PQ_2_111(P,Q)
            control = bool(sheet_spotload.cell(i, 8).value)
            status = bool(sheet_spotload.cell(i, 9).value)
            outage = bool(sheet_spotload.cell(i, 10).value)

            n_load += 1
            load_instance = InfoLoad(name, node, ZIP, conn, np.array(P), np.array(Q), phases, control, status, outage)
            load_list.append(load_instance)
            load_set.add(name)
            load_dict[name] = {"Node": node,
                               "ZIP": ZIP,
                               "Connection": conn,
                               "P": np.array([0,0,0]),
                               "Q": np.array(Q),
                               "Phases": phases,
                               "Control": control,
                               "Status": status,
                               "Outage": outage}
    return n_load, load_list, load_set, load_dict

def getdata_cap(sheet_cap):
    n_cap = 0
    cap_list = []
    cap_set = set([])
    cap_dict = {}

    data_row = 4
    for i in range(data_row, sheet_cap.nrows):
        if sheet_cap.cell(i, 0).value == 'Total':
            break
        else:
            node = num2str(sheet_cap.cell(i, 0).value)
            name = 'cap_' + node
            Q1 = read2float(sheet_cap.cell(i, 1).value)
            Q2 = read2float(sheet_cap.cell(i, 2).value)
            Q3 = read2float(sheet_cap.cell(i, 3).value)
            Q = [Q1, Q2, Q3]
            P = [0.0, 0.0, 0.0]
            phases = PQ_2_111(P,Q)
            control = bool(sheet_cap.cell(i,4).value)
            status = bool(sheet_cap.cell(i, 5).value)
            outage = bool(sheet_cap.cell(i, 6).value)

            n_cap += 1
            cap_instance = InfoCap(name, node, np.array(P), np.array(Q), phases, control, status, outage)
            cap_list.append(cap_instance)
            cap_set.add(name)
            cap_dict[name] = {"Node": node,
                              "P": P,
                              "Q": Q,
                              "Phases": phases,
                              "Control": control,
                              "Status": status,
                              "Outage": outage}
    return n_cap, cap_list, cap_set, cap_dict

def getdata_gen(sheet_gen):
    n_gen = 0
    gen_list = []
    gen_set = set([])
    gen_dict = {}

    data_row = 1
    for i in range(data_row, sheet_gen.nrows):
        node = num2str(sheet_gen.cell(i,0).value)
        name = 'gen_' + node
        type = sheet_gen.cell(i,1).value
        Capacity = read2float(sheet_gen.cell(i,2).value)
        Pmax = read2float(sheet_gen.cell(i,3).value)
        Pmin = read2float(sheet_gen.cell(i,4).value)
        Qmax = read2float(sheet_gen.cell(i,5).value)
        Qmin = read2float(sheet_gen.cell(i,6).value)
        Rmax = read2float(sheet_gen.cell(i,7).value)
        FRR = read2float(sheet_gen.cell(i,8).value)
        CUI = read2float(sheet_gen.cell(i,9).value)
        Phases = [1,1,1]  # assume all the DGs are three-phase DGs
        Control = bool(read2float(sheet_gen.cell(i, 10).value))
        Status = bool(read2float(sheet_gen.cell(i, 11).value))
        Outage = bool(read2float(sheet_gen.cell(i,12).value))

        n_gen += 1
        gen_instance = InfoGen(name, node, type, Capacity, Pmax, Pmin, Qmax, Qmin, Rmax, FRR, CUI, Phases, Control, Status, Outage)
        gen_list.append(gen_instance)
        gen_set.add(name)
        gen_dict[name] = {"Name": name,
                          "Node": node,
                          "Type": type,
                          "Pcap": Capacity,
                          "Pmax": Pmax,
                          "Pmin": Pmin,
                          "Qmax": Qmax,
                          "Qmin": Qmin,
                          "Rmax": Rmax,
                          "FRR":  FRR,
                          "CUI":  CUI,
                          "Phases": Phases,
                          "Control": Control,
                          "Status": Status,
                          "Outage": Outage}
    return n_gen, gen_list, gen_set, gen_dict

def getdata_ess(sheet_ess):
    n_ess = 0
    ess_list = []
    ess_set = set([])
    ess_dict = {}

    data_row = 1
    for i in range(data_row, sheet_ess.nrows):
        node = num2str(sheet_ess.cell(i,0).value)
        name = 'ess_' + node
        Phases   = ABC_2_111(sheet_ess.cell(i,1).value)
        Capacity = read2float(sheet_ess.cell(i,2).value)
        SOC_init = read2float(sheet_ess.cell(i,3).value)
        SOC_min  = read2float(sheet_ess.cell(i,4).value)
        SOC_max  = read2float(sheet_ess.cell(i,5).value)
        P_rated  = read2float(sheet_ess.cell(i,6).value)
        P_ch_min = read2float(sheet_ess.cell(i,7).value)
        P_ch_max = read2float(sheet_ess.cell(i,8).value)
        P_disch_min = read2float(sheet_ess.cell(i,9).value)
        P_disch_max = read2float(sheet_ess.cell(i,10).value)
        effi_ch     = read2float(sheet_ess.cell(i,11).value)
        effi_disch  = read2float(sheet_ess.cell(i,12).value)
        idle_P_loss = read2float(sheet_ess.cell(i,13).value)
        idle_Q_loss = read2float(sheet_ess.cell(i,14).value)
        Control = bool(read2float(sheet_ess.cell(i, 15).value))
        Status  = bool(read2float(sheet_ess.cell(i, 16).value))
        Outage  = bool(read2float(sheet_ess.cell(i, 17).value))

        n_ess += 1
        ess_instance = InfoESS(name, node, Phases, Capacity, SOC_init, SOC_min, SOC_max, P_rated, P_ch_min,
                               P_ch_max, P_disch_min, P_disch_max, effi_ch, effi_disch, idle_P_loss, idle_Q_loss,
                               Control, Status, Outage)
        ess_list.append(ess_instance)
        ess_set.add(name)
        ess_dict[name] = {"Name":name,
                          "Node":node,
                          "Phases":Phases,
                          "Capacity":Capacity,
                          "SOC_init":SOC_init,
                          "SOC_min":SOC_min,
                          "SOC_max":SOC_max,
                          "P_rated":P_rated,
                          "P_ch_min":P_ch_min,
                          "P_ch_max":P_ch_max,
                          "P_disch_min":P_disch_min,
                          "P_disch_max":P_disch_max,
                          "effi_ch":effi_ch,
                          "effi_disch":effi_disch,
                          "idle_P_loss":idle_P_loss,
                          "idle_Q_loss":idle_Q_loss,
                          "Control":Control,
                          "Status":Status,
                          "Outage":Outage}

    return n_ess, ess_list, ess_set, ess_dict



# def getdata_lineconfig_overhead(sheet_lineconfig):
#     title_row = title_row_location(sheet_lineconfig, 'Overhead Line', 'config', 'phas')
#     config_col = col_location(sheet_lineconfig, title_row, 'Overhead Line', 'config')
#     phasing_col = col_location(sheet_lineconfig, title_row, 'Overhead Line', 'phasing')
#     phasecond_col = col_location(sheet_lineconfig, title_row, 'Overhead Line', 'phase cond')
#     neutralcond_col = col_location(sheet_lineconfig, title_row, 'Overhead Line', 'neutral cond')
#     spacing_col = col_location(sheet_lineconfig, title_row, 'Overhead Line', 'spacing')
#
#     n_lineconfig = sheet_lineconfig.nrows - title_row - 2
#     lineconfig_dict = {}     # Not sure which structure is better: dict or list of objects?
#     lineconfig_set = set([])
#
#
#     # Starting from the row below the found title_row
#     for i in range(title_row + 2, sheet_lineconfig.nrows):
#         config = num2str(sheet_lineconfig.cell(i, config_col).value)
#         phasing = num2str(sheet_lineconfig.cell(i, phasing_col).value)
#         phasecond = num2str(sheet_lineconfig.cell(i, phasecond_col).value)
#         neutralcond = num2str(sheet_lineconfig.cell(i, neutralcond_col).value)
#         spacing = num2str(sheet_lineconfig.cell(i, spacing_col).value)
#         phase_material = num2str(sheet_lineconfig.cell(title_row + 1, phasecond_col).value)
#         neutral_material = num2str(sheet_lineconfig.cell(title_row + 1, neutralcond_col).value)
#
#         lineconfig_dict[config] = {"Phasing": phasing,
#                                    "Phase Conductor": phasecond,
#                                    "Phase Material": phase_material,
#                                    "Neutral Conductor": neutralcond,
#                                    "Neutral Material": neutral_material,
#                                    "Spacing": spacing,
#                                    "Geo": "Overhead"}
#         lineconfig_set.add(config)
#
#     return n_lineconfig, lineconfig_set, lineconfig_dict
#
# def getdata_lineconfig_undergnd(sheet_lineconfig):
#     title_row = title_row_location(sheet_lineconfig, 'Underground Line', 'config', 'phas')
#     config_col = col_location(sheet_lineconfig, title_row, 'Underground Line', 'config')
#     phasing_col = col_location(sheet_lineconfig, title_row, 'Underground Line', 'phasing')
#     cable_col = col_location(sheet_lineconfig, title_row, 'Underground Line', 'cable')
#     spacing_col = col_location(sheet_lineconfig, title_row, 'Underground Line', 'spacing')
#
#     n_lineconfig = sheet_lineconfig.nrows - title_row - 1
#     lineconfig_dict = {}     # Not sure which structure is better: dict or list of objects?
#     lineconfig_set = set([])
#
#     # Starting from the row below the found title_row
#     for i in range(title_row + 1, sheet_lineconfig.nrows):
#         config = num2str(sheet_lineconfig.cell(i, config_col).value)
#         phasing = num2str(sheet_lineconfig.cell(i, phasing_col).value)
#         cable = num2str(sheet_lineconfig.cell(i, cable_col).value)
#         spacing = num2str(sheet_lineconfig.cell(i, spacing_col).value)
#
#         lineconfig_dict[config] = {"Phasing": phasing,
#                                    "Cable": cable,
#                                    "Spacing": spacing,
#                                    "Geo": "Undergnd"}
#         lineconfig_set.add(config)
#
#     return n_lineconfig, lineconfig_set, lineconfig_dict
#
# def getdata_transformerconfig(sheet_transformer):
#     title_row = title_row_location(sheet_transformer, 'Transformer', 'kva', 'low')
#     config_col = 0
#     kva_col = col_location(sheet_transformer, title_row, 'Transformer', 'kva')
#     high_col = col_location(sheet_transformer, title_row, 'Transformer', 'high')
#     low_col = col_location(sheet_transformer, title_row, 'Transformer', 'low')
#     r_col = col_location(sheet_transformer, title_row, 'Transformer', 'r - %')
#     x_col = col_location(sheet_transformer, title_row, 'Transformer', 'x - %')
#
#     n_transformer = sheet_transformer.nrows - title_row - 1
#     transformer_dict = {}     # Not sure which structure is better: dict or list of objects?
#     transformer_set = set([])
#
#     # Starting from the row below the found title_row
#     for i in range(title_row + 1, sheet_transformer.nrows):
#         config = num2str(sheet_transformer.cell(i, config_col).value).replace(' ', '')
#         kva = num2str(sheet_transformer.cell(i, kva_col).value)
#         high = num2str(sheet_transformer.cell(i, high_col).value)
#         low = num2str(sheet_transformer.cell(i, low_col).value)
#         r = num2str(sheet_transformer.cell(i, r_col).value)
#         x = num2str(sheet_transformer.cell(i, x_col).value)
#
#         transformer_dict[config] = {"kVA": kva, "High Volt": high,
#                                    "Low Volt": low, "R%": r, "X%": x}
#
#         transformer_set.add(config)
#
#     return n_transformer, transformer_set, transformer_dict


def merge_components(d_line, d_switch, d_regulator):
    n_edge = d_line[0]
    list_edge = d_line[1]
    set_edge = d_line[2]   # line as a set
    dict_edge = d_line[3]

    n_switch = d_switch[0] # n_switch
    list_switch = d_switch[1]
    set_switch = d_switch[2]
    dict_switch = d_switch[3]

    for i in range(n_switch):
        sw_name = list_switch[i].Name
        new_line_name = sw_name.replace('switch', 'line')
        if new_line_name in set_edge:
            continue
        else:
            n_edge += 1
            new_dict = dict_switch[sw_name]
            add_switch = InfoLine(new_line_name,
                                  new_dict['Node_A'],
                                  new_dict['Node_B'],
                                  1,        # length
                                  'switch', # configuration
                                  new_dict['Impedance'],
                                  new_dict['Ampacity'],
                                  new_dict['Phases'],
                                  new_dict['Control'],
                                  new_dict['Status'],
                                  new_dict['Outage'])
            list_edge.append(add_switch)
            set_edge.add(new_line_name)
            dict_edge[new_line_name] = {"Node_A": add_switch.Node_A,
                                        "Node_B": add_switch.Node_B,
                                        "Config": 'switch',
                                        "Length": add_switch.Length_ft,
                                        "Impedance": add_switch.Impedance,
                                        "Ampacity": add_switch.Ampacity,
                                        "Phases": add_switch.Phases,
                                        "Control": add_switch.Control,
                                        "Status": add_switch.Status,
                                        "Outage": add_switch.Outage }

    n_rg = d_regulator[0]
    list_rg = d_regulator[1]
    set_rg = d_regulator[2]
    dict_rg = d_regulator[3]

    for i in range(n_rg):
        from_to = list_rg[i].Line
        mid = from_to.find('-')
        from_node = from_to[:mid]
        to_node = from_to[(mid+1):]
        location = list_rg[i].Node
        if location == from_node:
            new_from_node = from_node
            new_mid_node = from_node + 'R'
            new_to_node = to_node
        else:
            new_from_node = to_node
            new_mid_node = to_node + 'R'
            new_to_node = from_node
        new_RG_name = 'line' + '_' + new_mid_node + '_' + new_from_node
        new_line_name = 'line' + '_' + new_mid_node + '_' + new_to_node

        # Remove original line, add two new lines
        for j in range(n_edge):
            if list_edge[j].Node_A == from_node and list_edge[j].Node_B == to_node:
                gone_row = j
            elif list_edge[j].Node_B == from_node and list_edge[j].Node_A == to_node:
                gone_row = j
        buff_list = list_edge[gone_row]
        buff_name = buff_list.Name
        buff_dict = dict_edge[buff_list.Name]

        n_edge -= 1
        list_edge.pop(gone_row)
        set_edge.discard(buff_name)
        dict_edge.pop(buff_name)

        buff_list.Name = new_line_name
        buff_list.Node_A = new_mid_node
        buff_list.Node_B = new_to_node
        buff_dict = buff_dict

        n_edge += 1
        list_edge.append(buff_list)
        set_edge.add(new_line_name)
        dict_edge[new_line_name] = buff_dict

        n_edge += 1
        buff_list = InfoLine(new_RG_name,
                             new_mid_node,
                             new_from_node,
                             10,
                             'regulator',
                             list_rg[i].Impedance,
                             list_rg[i].Ampacity,
                             list_rg[i].Phases,
                             list_rg[i].Control,
                             list_rg[i].Status,
                             list_rg[i].Outage)
        buff_list.Node = location
        list_edge.append(buff_list)
        set_edge.add(new_RG_name)
        buff_dict = {"Node_A": new_mid_node,
                     "Node_B": new_from_node,
                     "Node"  : location,
                     "Length": 10,
                     "Config": "regulator",
                     "Impedance": list_rg[i].Impedance,
                     "Ampacity": list_rg[i].Ampacity,
                     "Phases": list_rg[i].Phases,
                     "Control": list_rg[i].Control,
                     "Status": list_rg[i].Status,
                     "Outage": list_rg[i].Outage}
        dict_edge[new_RG_name] = buff_dict

    return n_edge, list_edge, set_edge, dict_edge


def merge_load_cap(d_load, d_cap):
    n_loadcap = d_load[0]
    list_loadcap = d_load[1]
    set_loadcap = d_load[2]
    dict_loadcap = d_load[3]

    n_cap = d_cap[0]
    list_cap = d_cap[1]

    for i in range(n_cap):
        n_loadcap += 1
        name = list_cap[i].Name
        node = list_cap[i].Node
        zip  = 'PQ'
        conn = 'Y'
        p    = np.array(list_cap[i].P)*-1
        q    = np.array(list_cap[i].Q)*-1
        phases = list_cap[i].Phases
        control = list_cap[i].Control
        status = list_cap[i].Status
        outage = list_cap[i].Outage

        loadcap_instance = InfoLoad(name, node, zip, conn, p, q, phases, control, status, outage)
        list_loadcap.append(loadcap_instance)
        set_loadcap.add(name)
        dict_loadcap[name]={"Node": node,
                            "ZIP": zip,
                            "Connection": conn,
                            "P": p,
                            "Q": q,
                            "Phases": phases,
                            "Control": control,
                            "Status": status,
                            "Outage": outage}
    return n_loadcap, list_loadcap, set_loadcap, dict_loadcap


# Merging configuration files
# def merge_configurations(d_edge, config_overhead, config_undergnd, config_transformer):
#     n_line = d_edge[0]
#     line_list = d_edge[1]
#     line_set = d_edge[2]
#     line_dict = d_edge[3]
#
#     ohconfig_set = config_overhead[1]
#     ohconfig_dict = config_overhead[2]
#
#     ugconfig_set = config_undergnd[1]
#     ugconfig_dict = config_undergnd[2]
#
#     txconfig_set = config_transformer[1]
#     txconfig_dict = config_transformer[2]
#
#     line_matrix = linecode_library().line_matrix
#
#     for i in range(n_line):
#         config = line_list[i].Config
#         if config in ohconfig_set:
#             line_list[i].Phases = ohconfig_dict[config]['Phasing']
#             line_list[i].PhaseConductor = ohconfig_dict[config]['Phase Conductor']
#             line_list[i].PhaseMaterial = ohconfig_dict[config]['Phase Material']
#             line_list[i].NeutralConductor = ohconfig_dict[config]['Neutral Conductor']
#             line_list[i].NeutralMaterial = ohconfig_dict[config]['Neutral Material']
#             line_list[i].Spacing = ohconfig_dict[config]['Spacing']
#             line_list[i].Geo = ohconfig_dict[config]['Geo']
#             line_list[i].Imp = line_matrix[config]
#             line_dict[line_list[i].Name].update({'Phases': ohconfig_dict[config]['Phasing'],
#                                             'Phase Conductor': ohconfig_dict[config]['Phase Conductor'],
#                                             'Phase Material' : ohconfig_dict[config]['Phase Material'],
#                                             'Neutral Conductor' : ohconfig_dict[config]['Neutral Conductor'],
#                                             'Neutral Material' : ohconfig_dict[config]['Neutral Material'],
#                                             'Spacing' : ohconfig_dict[config]['Spacing'],
#                                             'Geo' : ohconfig_dict[config]['Geo'],
#                                             'Imp' : line_matrix[config]})
#         elif config in ugconfig_set:
#             line_list[i].Phases = ugconfig_dict[config]['Phasing']
#             line_list[i].Cable = ugconfig_dict[config]['Cable']
#             line_list[i].Spacing = ugconfig_dict[config]['Spacing']
#             line_list[i].Geo = ugconfig_dict[config]['Geo']
#             line_list[i].Imp = line_matrix[config]
#             line_dict[line_list[i].Name].update({'Phases': ugconfig_dict[config]['Phasing'],
#                                             'Cable' : ugconfig_dict[config]['Phasing'],
#                                             'Spacing' : ugconfig_dict[config]['Spacing'],
#                                             'Geo' : ugconfig_dict[config]['Geo'],
#                                             'Imp': line_matrix[config]})
#         elif config in txconfig_set:
#             line_list[i].kVA = txconfig_dict[config]['kVA']
#             line_list[i].HighV = txconfig_dict[config]['High Volt']
#             line_list[i].LowV = txconfig_dict[config]['Low Volt']
#             line_list[i].R = txconfig_dict[config]['R%']
#             line_list[i].X = txconfig_dict[config]['X%']
#             line_list[i].Imp = line_matrix[config]
#             # Phases?
#             line_dict[line_list[i].Name].update({'Phases': 'A B C N',
#                                             'kVA' : txconfig_dict[config]['kVA'],
#                                             'High Volt' : txconfig_dict[config]['High Volt'],
#                                             'Low Volt': txconfig_dict[config]['Low Volt'],
#                                             'R%' : txconfig_dict[config]['R%'],
#                                             'X%': txconfig_dict[config]['X%'],
#                                             'Imp': line_matrix[config]})
#         elif config == 'switch':
#             line_list[i].Imp = line_matrix[config]
#             line_dict[line_list[i].Name].update({ 'Imp': line_matrix[config] })
#         elif config == 'regulator':
#             pass
#         else:
#             print("The line configuration " + config + " is not found in the database")
#
#     return n_line, line_list, line_set, line_dict


def title_row_location(sheet, object='Line', key1='key word', key2='key word'):
    # find title_row
    title_row = sheet.nrows + 1
    for row in range(sheet.nrows):
        j = '   '
        for i in range(4):
            j += str(sheet.cell(row, i).value)
        # find the title line
        if j.lower().find(key1)>0:
            if j.lower().find(key2) > 0:
                title_row = row
                break
    if title_row > sheet.nrows:
        # TODO Prompt the error message and stop running.
        print(object + ' Title Information Not Found in the selected file')
    return title_row

def col_location(sheet, title_row, object, key):
    col = sheet.ncols + 1

    for column in range(sheet.ncols):
        if sheet.cell(title_row, column).value.lower().find(key) > -1:
            col = column
            break
    if col > sheet.ncols:
        print(object + 'Information is missing')
    return col

def edge_naming(sheet, header, row, from_col, to_col):
    # Create a name, then create instance for each line
    from_node = sheet.cell(row, from_col).value
    from_type = sheet.cell(row, from_col).ctype

    to_node = sheet.cell(row, to_col).value
    to_type = sheet.cell(row, to_col).ctype

    # Naming rules: num + num --> small_num + big_num; Text + num --> Text in lower case + num
    # Text + Text --> Text with small ASCII of first letter in lower case comes first
    if from_type == 1 and to_type == 1:
        for iter in range(min(len(from_node), len(to_node))):
            if ord(from_node[iter].lower()) > ord(to_node[iter].lower()):
                mid_node = from_node
                from_node = to_node
                to_node = mid_node
                break
    elif from_type == 1 and to_type == 2:
        pass
    elif from_type == 2 and to_type == 1:
        mid_node = from_node
        from_node = to_node
        to_node = mid_node
    elif from_type == 2 and to_type == 2:
        if from_node > to_node:
            mid_node = from_node
            from_node = to_node
            to_node = mid_node
    else:
        print('Unsupported format found in line data document')

    if from_type == 1:  # Text
        from_node = from_node.lower()
    elif from_type == 2:  # Number
        from_node = str(int(from_node))

    if to_type == 1:  # Text
        to_node = to_node.lower()
    elif to_type == 2:  # Number
        to_node = str(int(to_node))

    name = header + '_' + from_node + '_' + to_node

    return name, from_node, to_node

def read2float(a):
    if a == '':
        return 0.0
    else:
        return float(a)

def isnumber(aString):
    try:
        int(aString)
        return True
    except:
        return False

def num2str(sample):
    if isnumber(sample):
        if float(sample) - int(sample) != 0:
            return str(sample)
        else:
            return str(int(sample))
    else:
        return str(sample)

def ABC_2_111(ABC):
    if 'A' in ABC.upper():
        ph_a = np.array([1, 0, 0])
    else:
        ph_a = np.array([0, 0, 0])
    if 'B' in ABC.upper():
        ph_b = np.array([0, 1, 0])
    else:
        ph_b = np.array([0, 0, 0])
    if 'C' in ABC.upper():
        ph_c = np.array([0, 0, 1])
    else:
        ph_c = np.array([0, 0, 0])
    re = ph_a + ph_b + ph_c

    return re

def PQ_2_111(P,Q):
    ph_p = []
    ph_q = []
    for i in range(3):
        ph_p.append(int(P[i] != 0))
    for i in range(3):
        ph_q.append(int(Q[i] != 0))
    c = np.array(ph_p) + np.array(ph_q)
    d = []
    for i in range(3):
        if c[i] > 0:
            d.append(1)
        else:
            d.append(0)
    return d






def xlsx_read_edge(fileNameLine = None, fileNameSwitch = None, fileNameRegulator = None):

    cwd = os.getcwd()
    #line_data_file_name = cwd + r'\xlsx_data\line data_py.xlsx'        # replace it with customer input path
    #switch_data_file_name = cwd + r'\xlsx_data\switch data_py.xlsx'    # replace it with customer input path
    #regulator_data_file_name = cwd + r'\xlsx_data\Regulator Data_py.xlsx' # replace it with customer input path
    line_data_file_name = cwd + r'/xlsx_data/line data_py.xlsx'        # replace it with customer input path
    switch_data_file_name = cwd + r'/xlsx_data/switch data_py.xlsx'    # replace it with customer input path
    regulator_data_file_name = cwd + r'/xlsx_data/Regulator Data_py.xlsx' # replace it with customer input path



    if fileNameLine is not None:
      line_data_file_name = fileNameLine


    if fileNameSwitch is not None:
      switch_data_file_name = fileNameSwitch


    if fileNameRegulator is not None:
      regulator_data_file_name = fileNameRegulator




    file_location_line_orig = line_data_file_name
    file_location_switch_orig = switch_data_file_name
    file_location_regulator = regulator_data_file_name


    [file_type_line, file_location_line] = file_location_correction(file_location_line_orig)
    [file_type_switch, file_location_switch] = file_location_correction(file_location_switch_orig)
    [file_type_regulator, file_location_regulator] = file_location_correction(file_location_regulator)

    file_info_line = File(file_type_line, file_location_line)
    file_info_switch = File(file_type_switch, file_location_switch)
    file_info_regulator = File(file_type_regulator, file_location_regulator)


    #  Import line data into the workbook
    workbook_line = open_workbook(file_info_line.file_location)
    sheet_line = sheets_selection(workbook_line)
    data_line = getdata_line(sheet_line)

    # Import switch data into the workbook
    # If the switching file is available, also add switches into the line model.
    # But some lines are duplicate, use line_set to determine if a line/switch is unique
    workbook_switch = open_workbook(file_info_switch.file_location)
    sheet_switch = sheets_selection(workbook_switch)
    data_switch = getdata_switch(sheet_switch)

    # Import regulator data. Note regulators are in-line model, to count the
    # voltage levels between both sides, additional lines should be added when
    # merging line data and regulator data.
    workbook_regulator = open_workbook(file_info_regulator.file_location)
    sheet_regulator = sheets_selection(workbook_regulator)
    data_regulator = getdata_regulator(sheet_regulator)


    # --------------- Step 1: Merge data for lines, switches, and regulators --------
    data_edge = merge_components(data_line, data_switch, data_regulator)

    # # --------------- Step 2: Adding configuration properties to line data ----------
    # data_edge = merge_configurations(data_edge, data_config_overhead, data_config_undergnd, data_config_transformer)

    return data_edge, data_line, data_switch, data_regulator


def construct_node(fileNameNode = None):

    cwd = os.getcwd()
    #node_data_file_name = cwd + r'\xlsx_data\node data_py.xlsx'        # replace it with customer input path
    node_data_file_name = cwd + r'/xlsx_data/node data_py.xlsx'        # replace it with customer input path


    if fileNameNode is not None :
      node_data_file_name = fileNameNode


    file_location_node_orig = node_data_file_name


    [file_type_node, file_location_node] = file_location_correction(file_location_node_orig)
    file_info_node = File(file_type_node, file_location_node)
    workbook_node = open_workbook(file_info_node.file_location)
    sheet_node = sheets_selection(workbook_node)

    GIS = {}
    data_row = 3
    for i in range(data_row, sheet_node.nrows):
        node = num2str(sheet_node.cell(i,0).value)
        latitude = float(sheet_node.cell(i,1).value)
        longitude = float(sheet_node.cell(i,2).value)
        GIS.update({node:([latitude, longitude])})


    data_edge = xlsx_read_edge()[0]
    n_node = 0
    node_list = []
    node_set = set([])
    node_dict = {}

    n_edge = data_edge[0]
    edge_list = data_edge[1]

    for i in range(n_edge):
        nodeA = edge_list[i].Node_A
        nodeB = edge_list[i].Node_B
        if nodeA in node_set:
            pass
        else:
            n_node += 1
            name = 'node_' + nodeA
            node = nodeA
            number = n_node - 1
            # Determine phase information based on all the edges connected to this node
            phases = np.array([0,0,0])
            for j in edge_list:
                if node == j.Node_A:
                    phases += np.array(j.Phases)
                if node == j.Node_B:
                    phases += np.array(j.Phases)
            phases = (phases > 0).astype(int)
            position = GIS[nodeA]

            node_int = InfoNode(name, node, number, phases, position)
            node_list.append(node_int)
            node_set.add(node)
            node_dict[node] = {'Name': name,
                               'Node' : nodeA,
                               'Number': number,
                               'Phases': phases,
                               'GIS': position}

        if nodeB in node_set:
            pass
        else:
            n_node += 1
            name = 'node_' + nodeB
            node = nodeB
            number = n_node - 1

            phases = np.array([0,0,0])
            for j in edge_list:
                if node == j.Node_A:
                    phases += np.array(j.Phases)
                if node == j.Node_B:
                    phases += np.array(j.Phases)
            phases = (phases > 0).astype(int)
            position = GIS[nodeB]

            node_int = InfoNode(name, node, number, phases, position)
            node_list.append(node_int)
            node_set.add(node)
            node_dict[node] = {'Name': name,
                               'Node' : nodeB,
                               'Number': number,
                               'Phases': phases,
                               'GIS': position}

    return n_node, node_list, node_set, node_dict


def xlsx_read_load(fileNameLoad =  None, fileNameCapacitor =  None):

    cwd = os.getcwd()
    #spotload_data_file_name = cwd + r'\xlsx_data\spot loads data_py.xlsx'        # replace it with customer input path
    #cap_data_file_name = cwd + r'\xlsx_data\cap data_py.xlsx'                    # replace it with customer input path
    spotload_data_file_name = cwd + r'/xlsx_data/spot loads data_py.xlsx'        # replace it with customer input path
    cap_data_file_name = cwd + r'/xlsx_data/cap data_py.xlsx'                    # replace it with customer input path


    if fileNameLoad is not None :
      spotload_data_file_name = fileNameLoad


    if fileNameCapacitor is not None :
      cap_data_file_name = fileNameCapacitor


    file_location_spotload = spotload_data_file_name
    file_location_cap = cap_data_file_name


    [file_type_spotload, file_location_spotload] = file_location_correction(file_location_spotload)
    [file_type_cap, file_location_cap] = file_location_correction(file_location_cap)

    file_info_spotload = File(file_type_spotload, file_location_spotload)
    file_info_cap = File(file_type_cap, file_location_cap)

    # Import load data
    workbook_spotload = open_workbook(file_info_spotload.file_location)
    sheet_spotload = sheets_selection(workbook_spotload)
    data_load = getdata_spotload(sheet_spotload)

    # Import capacitor data
    workbook_cap = open_workbook(file_info_cap.file_location)
    sheet_cap = sheets_selection(workbook_cap)
    data_cap = getdata_cap(sheet_cap)

    data_loadcap = merge_load_cap(data_load, data_cap)

    return data_loadcap, data_load, data_cap


def xlsx_read_gen(fileNameDG =  None):

    cwd = os.getcwd()
    #gen_data_file_name = cwd + r'\xlsx_data\DG_py.xlsx'        # replace it with customer input path
    gen_data_file_name = cwd + r'/xlsx_data/DG_py.xlsx'        # replace it with customer input path


    if fileNameDG is not None :
      gen_data_file_name = fileNameDG


    file_location_gen = gen_data_file_name


    [file_type_gen, file_location_gen] = file_location_correction(file_location_gen)
    file_info_gen = File(file_type_gen, file_location_gen)
    workbook_gen = open_workbook(file_info_gen.file_location)
    sheet_gen = sheets_selection(workbook_gen)
    data_gen = getdata_gen(sheet_gen)

    return data_gen



def xlsx_read_ess(fileNameESS =  None):

    cwd = os.getcwd()
    #ess_data_file_name = cwd + r'\xlsx_data\ESS_py.xlsx'        # replace it with customer input path
    ess_data_file_name = cwd + r'/xlsx_data/ESS_py.xlsx'        # replace it with customer input path


    if fileNameESS is not None :
      ess_data_file_name = fileNameESS


    file_location_ess = ess_data_file_name


    [file_type_ess, file_location_ess] = file_location_correction(file_location_ess)
    file_info_ess = File(file_type_ess, file_location_ess)
    workbook_ess = open_workbook(file_info_ess.file_location)
    sheet_ess = sheets_selection(workbook_ess)
    data_ess = getdata_ess(sheet_ess)

    return data_ess




def add_3D_dict(thedict, key1, key2, val):
    if key1 in thedict:
        thedict[key1].update({key2: val})
    else:
        thedict.update({key1: {key2: val}})
    return thedict



def data_read(fileNameNode=None , fileNameLine=None, fileNameSwitch=None, fileNameRegulator=None, fileNameLoad=None, fileNameCapacitor=None, fileNameDG=None, fileNameESS=None):

    [data_edge, data_line, data_switch, data_regulator] = xlsx_read_edge(fileNameLine = fileNameLine, fileNameSwitch = fileNameSwitch, fileNameRegulator = fileNameRegulator)
    data_node = construct_node(fileNameNode = fileNameNode)
    data_gen = xlsx_read_gen(fileNameDG = fileNameDG)
    [data_loadcap, data_load, data_cap] = xlsx_read_load(fileNameLoad = fileNameLoad, fileNameCapacitor = fileNameCapacitor)
    data_ess = xlsx_read_ess(fileNameESS = fileNameESS)

    n_edge = data_edge[0]
    edge_list = data_edge[1]
    edge_set = data_edge[2]
    edge_dict = data_edge[3]

    n_line = data_line[0]
    line_list = data_line[1]
    line_set = data_line[2]
    line_dict = data_line[3]

    n_switch = data_switch[0]
    switch_list = data_switch[1]
    switch_set = data_switch[2]
    switch_dict = data_switch[3]

    n_regulator = data_regulator[0]
    regulator_list = data_regulator[1]
    regulator_set = data_regulator[2]
    regulator_dict = data_regulator[3]

    n_node = data_node[0]
    node_list = data_node[1]
    node_set = data_node[2]
    node_dict = data_node[3]

    n_gen = data_gen[0]
    gen_list = data_gen[1]
    gen_set = data_gen[2]
    gen_dict = data_gen[3]

    n_loadcap = data_loadcap[0]
    loadcap_list = data_loadcap[1]
    loadcap_set = data_loadcap[2]
    loadcap_dict = data_loadcap[3]

    n_load = data_load[0]
    load_list = data_load[1]
    load_set = data_load[2]
    load_dict = data_load[3]

    n_cap = data_cap[0]
    cap_list = data_cap[1]
    cap_set = data_cap[2]
    cap_dict = data_cap[3]

    n_ess = data_ess[0]
    ess_list = data_ess[1]
    ess_set = data_ess[2]
    ess_dict = data_ess[3]

    # Add "Number" attributes to each edge, line, switch, regulator, gen, loadcap, load, cap, According to node.Number
    for i in range(n_edge):
        edge_list[i].Number_A = node_dict[edge_list[i].Node_A]['Number']
        edge_list[i].Number_B = node_dict[edge_list[i].Node_B]['Number']
        edge_list[i].Index = i
    for i in range(n_line):
        line_list[i].Number_A = node_dict[line_list[i].Node_A]['Number']
        line_list[i].Number_B = node_dict[line_list[i].Node_B]['Number']
        line_list[i].Index = i
    for i in range(n_switch):
        switch_list[i].Number_A = node_dict[switch_list[i].Node_A]['Number']
        switch_list[i].Number_B = node_dict[switch_list[i].Node_B]['Number']
        switch_list[i].Index = i
    for i in range(n_regulator):
        regulator_list[i].Number_A = node_dict[regulator_list[i].Node_A]['Number']
        regulator_list[i].Number_B = node_dict[regulator_list[i].Node_B]['Number']
        regulator_list[i].Index = i

    for i in range(n_gen):
        gen_list[i].Number = node_dict[gen_list[i].Node]['Number']
        gen_list[i].Index = i
    for i in range(n_loadcap):
        loadcap_list[i].Number = node_dict[loadcap_list[i].Node]['Number']
        loadcap_list[i].Index = i
    for i in range(n_load):
        load_list[i].Number = node_dict[load_list[i].Node]['Number']
        load_list[i].Index = i
    for i in range(n_cap):
        cap_list[i].Number = node_dict[cap_list[i].Node]['Number']
        cap_list[i].Index = i
    for i in range(n_ess):
        ess_list[i].Number = node_dict[ess_list[i].Node]['Number']
        ess_list[i].Index = i


    # Adding more items to dictionaries for convenience
    for i in edge_list:
        val = edge_dict[i.Name]
        val['Number_A'] = i.Number_A
        val['Number_B'] = i.Number_B
        val['Index'] = i.Index
        edge_dict = add_3D_dict(edge_dict, i.Node_A, i.Node_B, val)
        edge_dict = add_3D_dict(edge_dict, i.Node_B, i.Node_A, val)

    obj = [n_edge, edge_list, edge_set, edge_dict,
           n_line, line_list, line_set, line_dict,
           n_switch, switch_list, switch_set, switch_dict,
           n_regulator, regulator_list, regulator_set, regulator_dict,
           n_node, node_list, node_set, node_dict,
           n_gen, gen_list, gen_set, gen_dict,
           n_loadcap, loadcap_list, loadcap_set, loadcap_dict,
           n_load, load_list, load_set, load_dict,
           n_cap, cap_list, cap_set, cap_dict,
           n_ess, ess_list, ess_set, ess_dict]

    #pickle.dump(obj, open("output/data.dat", "wb"), True)
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    file_to_save = os.path.join(data_folder, "data.dat")
    pickle.dump(obj, open(file_to_save, "wb"), True)
    #pickle.dump(obj, open("C:/Chen/dist-restoration/output/data.dat", "wb"), True)



if __name__ == '__main__':

    # [data_edge, data_line, data_switch, data_regulator] = xlsx_read_edge()
    # data_node = construct_node()
    # [data_loadcap, data_load, data_cap] = xlsx_read_load()
    # data_gen = xlsx_read_gen()
    # data_ess = xlsx_read_ess()

    data_read()

    print('done')




