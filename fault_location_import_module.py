#!/user/bin/env python3
# -*- coding: utf-8 -*-

from xlrd import open_workbook
import numpy as np
import os

# xlrd is one of the packages to work with Excel files in Python.
# Detailed information and tutorial can be found at
# http://www.python-excel.org/

__author__ = "Bo Chen"
__copyright__ = "Copyright 2018, " \
                "The GMLC Project: A Closed-Loop Distribution System Restoration Tool" \
                " for Natural Disaster Recovery"
__license__ = "MIT"  # To be determined
__version__ = "1.0.1"
__maintainer__ = "Bo Chen"
__email__ = "bo.chen@anl.gov"
__status__ = "Prototype"
__date__ = "3/15/2018"

##############################################################################################
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
# Define the class
class InfoFI(object):
    def __init__(self, Name='NoName', Node_A='-1', Node_B='-1', length = 0, FI= 0, Status = 1000):
        self.Name = Name
        self.Node_A = Node_A
        self.Node_B = Node_B
        self.Length = length
        self.FI = FI
        self.Status = Status

class InfoNode(object):
    def __init__(self, Name, Node, Number, GIS=[0.0, 0.0], Area = '0'):
        self.Name = Name
        self.Node = Node
        self.Number = Number  # numbering nodes, index of the vector
        self.GIS = GIS
        self.Area = Area


##############################################################################################
def getdata_fi(sheet_fi):
    title_row = title_row_location(sheet_fi, 'fi', 'node', 'fi')
    node_A_col = col_location(sheet_fi, title_row, 'fi', 'node a')
    node_B_col = col_location(sheet_fi, title_row, 'fi', 'node b')
    length_col = col_location(sheet_fi, title_row, 'fi', 'length')
    FI_col     = col_location(sheet_fi, title_row, 'fi', 'installation') # in lower case
    status_col = col_location(sheet_fi, title_row, 'fi', 'direction')

    n_fi = sheet_fi.nrows - title_row - 1
    fi_list=[]
    fi_dict = {}     # Not sure which structure is better: dict or list of objects?
    fi_set = set([]) # collection of unique elements

    # Starting from the row below the found title_row, assign values
    for i in range(title_row + 1, sheet_fi.nrows):
        [fi_name, from_node, to_node] = edge_naming(sheet_fi, 'fi', i, node_A_col, node_B_col)
        fi_length = sheet_fi.cell(i,length_col).value
        fi_installation = int(sheet_fi.cell(i, FI_col).value)
        fi_status = int(sheet_fi.cell(i, status_col).value)

        fi_instance = InfoFI(fi_name, from_node, to_node, fi_length, fi_installation, fi_status)
        fi_list.append(fi_instance)
        fi_set.add(fi_name)

        fi_dict[fi_name] = {"Node_A": from_node,
                            "Node_B": to_node,
                            "Length": fi_length,
                            "FI": fi_installation,
                            "Status": fi_status}

    return n_fi, fi_list, fi_set, fi_dict


def title_row_location(sheet, object='fi', key1='key word', key2='key word'):
    # find title_row
    title_row = sheet.nrows + 1
    for row in range(sheet.nrows):
        j = '   '
        for i in range(4):
            j += str(sheet.cell(row, i).value)
        # find the title
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
        print('info required not found in xls file')
        col = 1000
    return col

def edge_naming(sheet, header, row, from_col, to_col):
    # Create a name, then create instance
    from_node = sheet.cell(row, from_col).value
    from_type = sheet.cell(row, from_col).ctype

    to_node = sheet.cell(row, to_col).value
    to_type = sheet.cell(row, to_col).ctype

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





def xlsx_read_fi():
    #cwd = os.getcwd()
    #file_location_fi_orig = cwd + '\\fault_indicator_data.xls'
    #file_location_fi_orig = cwd + '/data/fault_indicator_data.xls'

    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"data")
    file_location_fi_orig = os.path.join(data_folder, "fault_indicator_data.xls")

    [file_type_fi, file_location_fi] = file_location_correction(file_location_fi_orig)

    file_info_fi = File(file_type_fi, file_location_fi) # form a File object

    #  Import  data into the workbook
    workbook_fi = open_workbook(file_info_fi.file_location)
    sheet_fi = sheets_selection(workbook_fi)
    data_fi = getdata_fi(sheet_fi)

    return data_fi





def construct_node():
    #cwd = os.getcwd()
    #file_location_node_orig = cwd + '\\node data_123node.xls'
    #file_location_node_orig = cwd + '/data/node_data_123node.xls'

    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"data")
    file_location_node_orig = os.path.join(data_folder, "node_data_123node.xls")

    [file_type_node, file_location_node] = file_location_correction(file_location_node_orig)
    file_info_node = File(file_type_node, file_location_node)
    workbook_node = open_workbook(file_info_node.file_location)
    sheet_node = sheets_selection(workbook_node)


    GIS = {}
    AREA = {}
    data_row = 3
    for i in range(data_row, sheet_node.nrows):
        node = num2str(sheet_node.cell(i,0).value)
        latitude = float(sheet_node.cell(i,1).value)
        longitude = float(sheet_node.cell(i,2).value)
        area = num2str(sheet_node.cell(i,3).value)
        GIS.update({node:([latitude, longitude])})
        AREA.update({node:area})

    n_node = 0
    node_list = []
    node_set = set([])
    node_dict = {}

    data_fi = xlsx_read_fi()
    n_fi = data_fi[0]
    fi_list = data_fi[1]

    for i in range(n_fi):
        nodeA = fi_list[i].Node_A
        nodeB = fi_list[i].Node_B
        if nodeA in node_set:
            pass
        else:
            n_node += 1
            name = 'node_' + nodeA
            node = nodeA
            number = n_node - 1
            position = GIS[nodeA]
            area = AREA[nodeA]

            node_int = InfoNode(name, node, number, position, area)
            node_list.append(node_int)
            node_set.add(node)
            node_dict[node] = {'Name': name,
                               'Node' : nodeA,
                               'Number': number,
                               'GIS': position,
                               'Area': area}

        if nodeB in node_set:
            pass
        else:
            n_node += 1
            name = 'node_' + nodeB
            node = nodeB
            number = n_node - 1
            position = GIS[nodeB]
            area = AREA[nodeB]


            node_int = InfoNode(name, node, number, position, area)
            node_list.append(node_int)
            node_set.add(node)
            node_dict[node] = {'Name': name,
                               'Node' : nodeB,
                               'Number': number,
                               'GIS': position,
                               'Area': area}

    return n_node, node_list, node_set, node_dict



def xlsx_read_weather():
    #cwd = os.getcwd()
    #file_location_we_orig = cwd + '\\weather_data.xls'
    #file_location_we_orig = cwd + '/data/weather_data.xls'

    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"data")
    file_location_we_orig = os.path.join(data_folder, "weather_data.xls")


    [file_type_we, file_location_we] = file_location_correction(file_location_we_orig)
    file_info_we = File(file_type_we, file_location_we)
    workbook_we = open_workbook(file_info_we.file_location)
    sheet_we = sheets_selection(workbook_we)

    AREA = {}
    data_row = 3
    for i in range(data_row, sheet_we.nrows):
        area = num2str(sheet_we.cell(i,0).value)
        prob = float(sheet_we.cell(i,1).value)
        AREA.update({area:prob})

    return AREA


if __name__ == '__main__':
    data_fi = xlsx_read_fi()
    data_node = construct_node()
    data_weater = xlsx_read_weather()


    print('done')
