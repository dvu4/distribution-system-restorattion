#!/user/bin/env python3
# -*- coding: utf-8 -*-

" This is the program for formulating and solving the restoration problem. "

# [08/27/2017] Rethink the data structure, go back to revise/add info to various models
# [08/28/2017] - [08/31/2017] Formulate the SR problem, according to the MATLAB code
# [09/01/2017] Debugging the optimization model and verify the solution without power flow constraints
# [09/04/2017] Change the definition method for variables. Using addVars cannot be converted to np.array. Use [addVar for in] and reshape it.
# [09/05/2017] Found an undecleared default value when defining GUROBI variables:  m.addVar(). The default variable is ===non-negative=== !!!
# [09/10/2017] Debug infeasible solution caused by votlage model and constraints
# [09/11/2017] Debug infeasible solution caused by voltage regulator constraints
# [09/12/2017] Finish other constraints
# [09/13/2017] Add ESS and count into the rolling-horizon feature.
# [09/14/2017] Finish debugging CLPU load models under rolling-horizon condition
# [09/19/2017] Modified to adapt to the GUI interfaces
# [10/19/2018] Add os.path module to work around the different kinds of operating system-specific file system issues


__author__ = "Bo Chen"
__copyright__ = "Copyright 2017, " \
                "The GMLC Project: A Closed-Loop Distribution System Restoration Tool" \
                " for Natural Disaster Recovery"
__license__ = "MIT"  # To be determined
__version__ = "1.0.1"
__maintainer__ = "Bo Chen"
__email__ = "bo.chen@anl.gov"
__status__ = "Prototype"
__date__ = "08/27/2017"  # Starting date



import numpy as np
import data_import_module
import pickle
import os 
#from gurobipy import *



def data_import():
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    myfile = os.path.join(data_folder, "data.dat")
    obj = pickle.load(open(myfile, "rb"))
    #obj = pickle.load(open("output/data.dat", "rb"))
    
    n_edge = obj[0]
    edge_list = obj[1]
    edge_set = obj[2]
    edge_dict = obj[3]
    n_line = obj[4]
    line_list = obj[5]
    line_set = obj[6]
    line_dict = obj[7]
    n_switch = obj[8]
    switch_list = obj[9]
    switch_set = obj[10]
    switch_dict = obj[11]
    n_regulator = obj[12]
    regulator_list = obj[13]
    regulator_set = obj[14]
    regulator_dict = obj[15]
    n_node = obj[16]
    node_list = obj[17]
    node_set = obj[18]
    node_dict = obj[19]
    n_gen = obj[20]
    gen_list = obj[21]
    gen_set = obj[22]
    gen_dict = obj[23]
    n_loadcap = obj[24]
    loadcap_list = obj[25]
    loadcap_set = obj[26]
    loadcap_dict = obj[27]
    n_load = obj[28]
    load_list = obj[29]
    load_set = obj[30]
    load_dict = obj[31]
    n_cap = obj[32]
    cap_list = obj[33]
    cap_set = obj[34]
    cap_dict = obj[35]
    n_ess = obj[36]
    ess_list = obj[37]
    ess_set = obj[38]
    ess_dict = obj[39]

    return n_edge, edge_list, edge_set, edge_dict, \
           n_line, line_list, line_set, line_dict,\
           n_switch, switch_list, switch_set, switch_dict,\
           n_regulator, regulator_list, regulator_set, regulator_dict,\
           n_node, node_list, node_set, node_dict,\
           n_gen, gen_list, gen_set, gen_dict,\
           n_loadcap, loadcap_list, loadcap_set, loadcap_dict,\
           n_load, load_list, load_set, load_dict,\
           n_cap, cap_list, cap_set, cap_dict,\
           n_ess, ess_list, ess_set, ess_dict

def solve(rh_setup, sr_setup):

    def impedance_matrix(n_edge, n_node, edge_list, Zbase):
        Z = np.zeros((3 * n_node, 3 * n_node)) * 1j
        for i in range(n_edge):
            Nfrom = edge_list[i].Number_A + 1
            Nto = edge_list[i].Number_B + 1
            Imp = edge_list[i].Impedance/Zbase  # impedance per mile
            Z[(3 * Nfrom - 2 - 1):(3 * Nfrom), (3 * Nto - 2 - 1):(3 * Nto)] = Imp
            Z[(3 * Nto - 2 - 1):(3 * Nto), (3 * Nfrom - 2 - 1):3 * Nfrom] = Imp
        return Z

    class graph_node(object):
        def __init__(self, name):
            self.__name = name
            self.__links = set()

        @property
        def name(self):
            return self.__name

        @property
        def links(self):
            return set(self.__links)

        def add_link(self, other):
            self.__links.add(other)
            other.__links.add(self)

    def connected_components(nodes):
        # List of connected components found. The order is random.
        result = []

        # Make a copy of the set, so we can modify it.
        nodes = set(nodes)

        # Iterate while we still have nodes to process.
        while nodes:

            # Get a random node and remove it from the global set.
            n = nodes.pop()

            # This set will contain the next group of nodes connected to each other.
            group = {n}

            # Build a queue with this node in it.
            queue = [n]

            # Iterate the queue.
            # When it's empty, we finished visiting a group of connected nodes.
            while queue:
                # Consume the next item from the queue.
                n = queue.pop(0)

                # Fetch the neighbors.
                neighbors = n.links

                # Remove the neighbors we already visited.
                neighbors.difference_update(group)

                # Remove the remaining nodes from the global set.
                nodes.difference_update(neighbors)

                # Add them to the group of connected nodes.
                group.update(neighbors)

                # Add them to the queue, so we visit them in the next iterations.
                queue.extend(neighbors)

            # Add the group to the list of groups.
            result.append(group)

        # Return the list of groups.
        return result

    def find_islands(node_dict, node_set, edge_dict):
        # Thanks to the author of the algorthm: https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph/
        graph_node_set = set()
        graph_dict = node_dict.copy()  # pay extra attention here
        for i in graph_dict:
            graph_dict[i] = graph_node(node_dict[i]["Node"])
            graph_node_set.add(graph_dict[i])
        for i in edge_dict:
            if i in node_set:
                for j in edge_dict[i]:
                    if edge_dict[i][j]["Control"] == False:
                        graph_dict[i].add_link(graph_dict[j])
        graph_islands = connected_components(graph_node_set)
        islands = []
        for i in graph_islands:
            subislands = set()
            for j in i:
                subislands.update([j.name])  # not j.name, must be [j.name] as a string
            islands.append(subislands)

        return islands

    class solution(object):
        def __init__(self):
            pass

    ######################################################################################
    rh_start_time  = rh_setup['rh_start_time']
    rh_horizon     = rh_setup['rh_horizon']  # total steps in each iteration
    rh_control     = rh_setup['rh_control']  # within each iteration, how many steps to carry out
    rh_set_step    = rh_setup['rh_set_step']  # steps set by the user
    rh_step_length = rh_setup['rh_step_length']
    rh_iteration   = rh_setup['rh_iteration']
    rh_total_step  = rh_setup['rh_total_step'] # total steps used by the algorithm

    sr_clpu_enable    = sr_setup['sr_clpu_enable']  # enable Cold Load Pick Up load model
    sr_re_enable      = sr_setup['sr_re_enable']  # enable considering renewable energies
    sr_es_enable      = sr_setup['sr_es_enable']  # enable considering ESS model
    sr_rg_enable      = sr_setup['sr_rg_enable']  # enable considering voltage regulator
    sr_Vbase          = sr_setup['sr_Vbase']
    sr_Sbase          = sr_setup['sr_Sbase']
    sr_cap_enable     = sr_setup['sr_cap_enable'] # enable considering capacitor bank
    sr_n_polygon      = sr_setup['sr_n_polygon']  # number of polygen to approximate x^2 + y^2 <= C
    sr_Vsrc           = sr_setup['sr_Vsrc']  # expected voltage of the black-start DG
    sr_M              = sr_setup['sr_M']   # value used in the big-M method.
    sr_reserve_margin = sr_setup['sr_reserve_margin']

    ######################################################################################
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

    #####################################################################################
    #####################################################################################
    # ------------------------ Testing Area ------------------------- #
    # ------------ Manually set up initial conditions --------------- #
    # for i in range(n_edge):
    #     edge_list[i].Control = True  # all lines are controllable remotely
    #     edge_dict[edge_list[i].Name]['Control'] = True
    #     edge_list[i].Status = False  # disconnect all the lines
    #     edge_dict[edge_list[i].Name]['Status'] = False
    #     edge_list[i].Outage = False  # no outaged lines
    #     edge_dict[edge_list[i].Name]['Outage'] = False
    #
    # for i in loadcap_list:
    #     i.Control = True

    # for i in gen_list:
    #     if i.Node == '13':
    #         i.Type = 'black start'  # assign only one black start DG on node xx.
    #         gen_dict[i.Name]["Type"] = 'black start'
    #     else:
    #         i.Type = 'non black start'  # All the other DGs are non black start DG
    #         gen_dict[i.Name]["Type"] = 'non black start'

    # TODO: ESS power factor: if power factor is NAN, ESS output active power only. Otherwise, output P and Q according to the power factor

    for i in range(n_edge):
        if edge_list[i].Config == 'regulator':
            if edge_list[i].Node == '150':
                edge_list[i].Position = np.array([7*0.00625+1, 1, 1])
            if edge_list[i].Node == '160':
                edge_list[i].Position = [8*0.00625+1, 1*0.00625+1, 5*0.00625+1]
            if edge_list[i].Node == '25':
                edge_list[i].Position = [1, 0, -1*0.00625+1]
            if edge_list[i].Node == '9':
                edge_list[i].Position = [-1*0.00625+1, 0, 0]
    ######################################################################################

    ######################################################################################
    # Develop Nodal Impedance Matrix
    Zbase = (sr_Vbase/(3**0.5))**2/sr_Sbase
    Z = impedance_matrix(n_edge, n_node, edge_list, Zbase)

    # Develop Incident Matrices
    I_gen_node = np.zeros((n_gen, n_node))
    for i in range(n_gen):
        for j in range(n_node):
            if gen_list[i].Number == node_list[j].Number:
                I_gen_node[i,j] = 1
            else:
                I_gen_node[i,j] = 0

    I_loadcap_node = np.zeros((n_loadcap, n_node))
    for i in range(n_loadcap):
        for j in range(n_node):
            if loadcap_list[i].Number == node_list[j].Number:
                I_loadcap_node[i,j] = -1
            else:
                I_loadcap_node[i,j] = 0

    I_edge_node = np.zeros((n_edge, n_node))
    for i in range(n_edge):
        for j in range(n_node):
            if edge_list[i].Number_A == node_list[j].Number:
                I_edge_node[i,j] = -1  # leave "FROM" node
            elif edge_list[i].Number_B == node_list[j].Number:
                I_edge_node[i,j] = 1   # inject into "TO" node
            else:
                I_edge_node[i,j] = 0

    I_ess_node = np.zeros((n_ess, n_node))
    for i in range(n_ess):
        for j in range(n_node):
            if ess_list[i].Number == node_list[j].Number:
                I_ess_node[i, j] = 1
            else:
                I_ess_node[i, j] = 0

    # Group directly connected nodes and lines based on the graph determined by edge_list[i].Control
    islands = find_islands(node_dict, node_set, edge_dict)

    ######################################################################################
    # set up constraints to be included
    st_pf_balance        = rh_setup['rh_model_config']['st_pf_balance']
    st_pf_voltage        = rh_setup['rh_model_config']['st_pf_voltage']
    st_voltage_limit     = rh_setup['rh_model_config']['st_voltage_limit']
    st_line_capacity     = rh_setup['rh_model_config']['st_line_capacity']
    st_gen_capacity      = rh_setup['rh_model_config']['st_gen_capacity']
    st_gen_reserve       = rh_setup['rh_model_config']['st_gen_reserve']
    st_gen_ramp          = rh_setup['rh_model_config']['st_gen_ramp']
    st_gen_stepload      = rh_setup['rh_model_config']['st_gen_stepload']
    st_gen_unbalance     = rh_setup['rh_model_config']['st_gen_unbalance']

    st_non_shed      = rh_setup['rh_model_config']['st_non_shed']
    st_connectivity  = rh_setup['rh_model_config']['st_connectivity']
    st_sequence      = rh_setup['rh_model_config']['st_sequence']
    st_topology      = rh_setup['rh_model_config']['st_topology']

    st_init          = True

    ######################################################################################
    ######################################################################################
    for iteration in range(rh_iteration):
        sr = Model('MILP')
        sr.reset()
        # Define decision variables using GUROBI Python language
        # Variables for unbalanced condition are not defined at this stage

        # Regulators ratios ranging from 0.9 to 1.1
        VG_A_ratio2= [sr.addVar(vtype=GRB.CONTINUOUS, name="VG_A_ratio2["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_regulator)]
        VG_A_ratio2 = np.array(VG_A_ratio2).reshape((rh_horizon, n_regulator))
        VG_B_ratio2= [sr.addVar(vtype=GRB.CONTINUOUS, name="VG_B_ratio2["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_regulator)]
        VG_B_ratio2 = np.array(VG_B_ratio2).reshape((rh_horizon, n_regulator))
        VG_C_ratio2= [sr.addVar(vtype=GRB.CONTINUOUS, name="VG_C_ratio2["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_regulator)]
        VG_C_ratio2 = np.array(VG_C_ratio2).reshape((rh_horizon, n_regulator))

        # generators
        x_gen   = [sr.addVar(vtype=GRB.BINARY, name="x_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        x_gen   = np.array(x_gen).reshape((rh_horizon, n_gen))
        P_A_gen = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_A_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        P_A_gen = np.array(P_A_gen).reshape((rh_horizon, n_gen))
        P_B_gen = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_B_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        P_B_gen = np.array(P_B_gen).reshape((rh_horizon, n_gen))
        P_C_gen = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_C_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        P_C_gen = np.array(P_C_gen).reshape((rh_horizon, n_gen))
        Q_A_gen = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_A_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        Q_A_gen = np.array(Q_A_gen).reshape((rh_horizon, n_gen))
        Q_B_gen = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_B_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        Q_B_gen = np.array(Q_B_gen).reshape((rh_horizon, n_gen))
        Q_C_gen = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_C_gen["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        Q_C_gen = np.array(Q_C_gen).reshape((rh_horizon, n_gen))

        # loads
        x_load   = [sr.addVar(vtype=GRB.BINARY, name="x_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        x_load   = np.array(x_load).reshape((rh_horizon, n_loadcap))
        P_A_load = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_A_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        P_A_load = np.array(P_A_load).reshape((rh_horizon, n_loadcap))
        P_B_load = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_B_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        P_B_load = np.array(P_B_load).reshape((rh_horizon, n_loadcap))
        P_C_load = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_C_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        P_C_load = np.array(P_C_load).reshape((rh_horizon, n_loadcap))
        Q_A_load = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_A_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        Q_A_load = np.array(Q_A_load).reshape((rh_horizon, n_loadcap))
        Q_B_load = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_B_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        Q_B_load = np.array(Q_B_load).reshape((rh_horizon, n_loadcap))
        Q_C_load = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_C_load["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_loadcap)]
        Q_C_load = np.array(Q_C_load).reshape((rh_horizon, n_loadcap))

        # edges
        x_edge   = [sr.addVar(vtype=GRB.BINARY, name="x_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        x_edge   = np.array(x_edge).reshape((rh_horizon, n_edge))
        P_A_edge = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_A_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        P_A_edge = np.array(P_A_edge).reshape((rh_horizon, n_edge))
        P_B_edge = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_B_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        P_B_edge = np.array(P_B_edge).reshape((rh_horizon, n_edge))
        P_C_edge = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_C_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        P_C_edge = np.array(P_C_edge).reshape((rh_horizon, n_edge))
        Q_A_edge = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_A_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        Q_A_edge = np.array(Q_A_edge).reshape((rh_horizon, n_edge))
        Q_B_edge = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_B_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        Q_B_edge = np.array(Q_B_edge).reshape((rh_horizon, n_edge))
        Q_C_edge = [sr.addVar(lb= -1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_C_edge["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_edge)]
        Q_C_edge = np.array(Q_C_edge).reshape((rh_horizon, n_edge))

        # nodes
        x_node   = [sr.addVar(vtype=GRB.BINARY, name="x_node["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_node)]
        x_node   = np.array(x_node).reshape((rh_horizon, n_node))
        V2_A_node= [sr.addVar(vtype=GRB.CONTINUOUS, name="V2_A_node["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_node)]
        V2_A_node = np.array(V2_A_node).reshape((rh_horizon, n_node))
        V2_B_node= [sr.addVar(vtype=GRB.CONTINUOUS, name="V2_B_node["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_node)]
        V2_B_node = np.array(V2_B_node).reshape((rh_horizon, n_node))
        V2_C_node= [sr.addVar(vtype=GRB.CONTINUOUS, name="V2_C_node["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_node)]
        V2_C_node = np.array(V2_C_node).reshape((rh_horizon, n_node))

        # ESS
        x_ess_ch = [sr.addVar(vtype=GRB.BINARY, name="x_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        x_ess_ch   = np.array(x_ess_ch).reshape((rh_horizon, n_ess))
        x_ess_disch = [sr.addVar(vtype=GRB.BINARY, name="x_ess_disch[" + str(t) + "," + str(e) + "]") for t in range(rh_horizon) for e in range(n_ess)]
        x_ess_disch = np.array(x_ess_disch).reshape((rh_horizon, n_ess))
        SOC_A = [sr.addVar(vtype=GRB.CONTINUOUS, name="SOC_A["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        SOC_A = np.array(SOC_A).reshape((rh_horizon, n_ess))
        SOC_B = [sr.addVar(vtype=GRB.CONTINUOUS, name="SOC_B["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        SOC_B = np.array(SOC_B).reshape((rh_horizon, n_ess))
        SOC_C = [sr.addVar(vtype=GRB.CONTINUOUS, name="SOC_C["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        SOC_C = np.array(SOC_C).reshape((rh_horizon, n_ess))
        P_A_ess_ch = [sr.addVar(vtype=GRB.CONTINUOUS, name="P_A_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        P_A_ess_ch = np.array(P_A_ess_ch).reshape((rh_horizon, n_ess))
        P_B_ess_ch = [sr.addVar(vtype=GRB.CONTINUOUS, name="P_B_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        P_B_ess_ch = np.array(P_B_ess_ch).reshape((rh_horizon, n_ess))
        P_C_ess_ch = [sr.addVar(vtype=GRB.CONTINUOUS, name="P_C_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        P_C_ess_ch = np.array(P_C_ess_ch).reshape((rh_horizon, n_ess))
        P_A_ess_disch = [sr.addVar(vtype=GRB.CONTINUOUS, name="P_A_ess_disch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        P_A_ess_disch = np.array(P_A_ess_disch).reshape((rh_horizon, n_ess))
        P_B_ess_disch = [sr.addVar(vtype=GRB.CONTINUOUS, name="P_B_ess_disch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        P_B_ess_disch = np.array(P_B_ess_disch).reshape((rh_horizon, n_ess))
        P_C_ess_disch = [sr.addVar(vtype=GRB.CONTINUOUS, name="P_C_ess_disch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        P_C_ess_disch = np.array(P_C_ess_disch).reshape((rh_horizon, n_ess))
        Q_A_ess_ch = [sr.addVar(vtype=GRB.CONTINUOUS, name="Q_A_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        Q_A_ess_ch = np.array(Q_A_ess_ch).reshape((rh_horizon, n_ess))
        Q_B_ess_ch = [sr.addVar(vtype=GRB.CONTINUOUS, name="Q_B_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        Q_B_ess_ch = np.array(Q_B_ess_ch).reshape((rh_horizon, n_ess))
        Q_C_ess_ch = [sr.addVar(vtype=GRB.CONTINUOUS, name="Q_C_ess_ch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        Q_C_ess_ch = np.array(Q_C_ess_ch).reshape((rh_horizon, n_ess))
        Q_A_ess_disch = [sr.addVar(vtype=GRB.CONTINUOUS, name="Q_A_ess_disch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        Q_A_ess_disch = np.array(Q_A_ess_disch).reshape((rh_horizon, n_ess))
        Q_B_ess_disch = [sr.addVar(vtype=GRB.CONTINUOUS, name="Q_B_ess_disch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        Q_B_ess_disch = np.array(Q_B_ess_disch).reshape((rh_horizon, n_ess))
        Q_C_ess_disch = [sr.addVar(vtype=GRB.CONTINUOUS, name="Q_C_ess_disch["+str(t)+","+str(e)+"]") for t in range(rh_horizon) for e in range(n_ess)]
        Q_C_ess_disch = np.array(Q_C_ess_disch).reshape((rh_horizon, n_ess))

        # DG Unbalance Variables
        ypn  = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ypn["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        ypn  = np.array(ypn).reshape((rh_horizon, n_gen))
        yqn  = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="yqn["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        yqn  = np.array(yqn).reshape((rh_horizon, n_gen))
        ypp  = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ypp["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        ypp  = np.array(ypp).reshape((rh_horizon, n_gen))
        yqp  = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="yqp["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        yqp  = np.array(yqp).reshape((rh_horizon, n_gen))
        dpn1 = [sr.addVar(vtype=GRB.BINARY, name="dpn1["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        dpn1 = np.array(dpn1).reshape((rh_horizon, n_gen))
        dpn2 = [sr.addVar(vtype=GRB.BINARY, name="dpn2["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        dpn2 = np.array(dpn2).reshape((rh_horizon, n_gen))
        dqn1 = [sr.addVar(vtype=GRB.BINARY, name="dqn1["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        dqn1 = np.array(dqn1).reshape((rh_horizon, n_gen))
        dqn2 = [sr.addVar(vtype=GRB.BINARY, name="dqn2["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        dqn2 = np.array(dqn2).reshape((rh_horizon, n_gen))
        dqp1 = [sr.addVar(vtype=GRB.BINARY, name="dqp1["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        dqp1 = np.array(dqp1).reshape((rh_horizon, n_gen))
        dqp2 = [sr.addVar(vtype=GRB.BINARY, name="dqp2["+str(t)+","+str(g)+"]") for t in range(rh_horizon) for g in range(n_gen)]
        dqp2 = np.array(dqp2).reshape((rh_horizon, n_gen))
        ynmax   = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ynmax["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        ynmax   = np.array(ynmax).reshape((rh_horizon, n_gen))
        ynmin   = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ynmin["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        ynmin   = np.array(ynmin).reshape((rh_horizon, n_gen))
        dnmax1 = [sr.addVar(vtype=GRB.BINARY, name="dnmax1[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dnmax1 = np.array(dnmax1).reshape((rh_horizon, n_gen))
        dnmax2 = [sr.addVar(vtype=GRB.BINARY, name="dnmax2[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dnmax2 = np.array(dnmax2).reshape((rh_horizon, n_gen))
        dnmin1 = [sr.addVar(vtype=GRB.BINARY, name="dnmin1[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dnmin1 = np.array(dnmin1).reshape((rh_horizon, n_gen))
        dnmin2 = [sr.addVar(vtype=GRB.BINARY, name="dnmin2[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dnmin2 = np.array(dnmin2).reshape((rh_horizon, n_gen))
        ypmax  = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ypmax["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        ypmax  = np.array(ypmax).reshape((rh_horizon, n_gen))
        ypmin  = [sr.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ypmin["+str(t)+"," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        ypmin  = np.array(ypmin).reshape((rh_horizon, n_gen))
        dpmax1 = [sr.addVar(vtype=GRB.BINARY, name="dpmax1[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dpmax1 = np.array(dpmax1).reshape((rh_horizon, n_gen))
        dpmax2 = [sr.addVar(vtype=GRB.BINARY, name="dpmax2[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dpmax2 = np.array(dpmax2).reshape((rh_horizon, n_gen))
        dpmin1 = [sr.addVar(vtype=GRB.BINARY, name="dpmin1[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dpmin1 = np.array(dpmin1).reshape((rh_horizon, n_gen))
        dpmin2 = [sr.addVar(vtype=GRB.BINARY, name="dpmin2[" + str(t) + "," + str(g) + "]") for t in range(rh_horizon) for g in range(n_gen)]
        dpmin2 = np.array(dpmin2).reshape((rh_horizon, n_gen))


        # renewable energies are not considered at this stage



        sr.update()

        ######################################################################################
        # Power balance constraints
        if st_pf_balance == True:
            # status according to the energization status
            for t in range(rh_horizon):
                for e in range(n_edge):
                    sr.addConstr(-1 * sr_M * x_edge[t, e] <= P_A_edge[t, e], "power_flow")
                    sr.addConstr( 1 * sr_M * x_edge[t, e] >= P_A_edge[t, e], "power_flow")
                    sr.addConstr(-1 * sr_M * x_edge[t, e] <= P_B_edge[t, e], "power_flow")
                    sr.addConstr( 1 * sr_M * x_edge[t, e] >= P_B_edge[t, e], "power_flow")
                    sr.addConstr(-1 * sr_M * x_edge[t, e] <= P_C_edge[t, e], "power_flow")
                    sr.addConstr( 1 * sr_M * x_edge[t, e] >= P_C_edge[t, e], "power_flow")
                    sr.addConstr(-1 * sr_M * x_edge[t, e] <= Q_A_edge[t, e], "power_flow")
                    sr.addConstr( 1 * sr_M * x_edge[t, e] >= Q_A_edge[t, e], "power_flow")
                    sr.addConstr(-1 * sr_M * x_edge[t, e] <= Q_B_edge[t, e], "power_flow")
                    sr.addConstr( 1 * sr_M * x_edge[t, e] >= Q_B_edge[t, e], "power_flow")
                    sr.addConstr(-1 * sr_M * x_edge[t, e] <= Q_C_edge[t, e], "power_flow")
                    sr.addConstr( 1 * sr_M * x_edge[t, e] >= Q_C_edge[t, e], "power_flow")
                    if edge_list[e].Phases[0] == 0:
                        sr.addConstr(0 == P_A_edge[t, e], "power_flow")
                        sr.addConstr(0 == Q_A_edge[t, e], "power_flow")
                    if edge_list[e].Phases[1] == 0:
                        sr.addConstr(0 == P_B_edge[t, e], "power_flow")
                        sr.addConstr(0 == Q_B_edge[t, e], "power_flow")
                    if edge_list[e].Phases[2] == 0:
                        sr.addConstr(0 == P_C_edge[t, e], "power_flow")
                        sr.addConstr(0 == Q_C_edge[t, e], "power_flow")
                for g in range(n_gen):
                    sr.addConstr(0 <= P_A_gen[t, g], "power_flow")
                    sr.addConstr(0 <= P_B_gen[t, g], "power_flow")
                    sr.addConstr(0 <= P_C_gen[t, g], "power_flow")
                    sr.addConstr(sr_M * x_gen[t, g] >= P_A_gen[t, g], "power_flow")
                    sr.addConstr(sr_M * x_gen[t, g] >= P_B_gen[t, g], "power_flow")
                    sr.addConstr(sr_M * x_gen[t, g] >= P_C_gen[t, g], "power_flow")
                    sr.addConstr(-1 * sr_M * x_gen[t, g] <= Q_A_gen[t, g], "power_flow")
                    sr.addConstr(-1 * sr_M * x_gen[t, g] <= Q_B_gen[t, g], "power_flow")
                    sr.addConstr(-1 * sr_M * x_gen[t, g] <= Q_C_gen[t, g], "power_flow")
                    sr.addConstr(sr_M * x_gen[t, g] >= Q_A_gen[t, g], "power_flow")
                    sr.addConstr(sr_M * x_gen[t, g] >= Q_B_gen[t, g], "power_flow")
                    sr.addConstr(sr_M * x_gen[t, g] >= Q_C_gen[t, g], "power_flow")
                    if gen_list[g].Phases[0] == 0:
                        sr.addConstr(0 == P_A_gen[t, g], "power_flow")
                        sr.addConstr(0 == Q_A_gen[t, g], "power_flow")
                    if gen_list[g].Phases[1] == 0:
                        sr.addConstr(0 == P_B_gen[t, g], "power_flow")
                        sr.addConstr(0 == Q_B_gen[t, g], "power_flow")
                    if gen_list[g].Phases[2] == 0:
                        sr.addConstr(0 == P_C_gen[t, g], "power_flow")
                        sr.addConstr(0 == Q_C_gen[t, g], "power_flow")


            # power balance
            PF_BAL_PA = P_A_gen.dot(I_gen_node) + P_A_load.dot(I_loadcap_node) + P_A_edge.dot(I_edge_node) + P_A_ess_disch.dot(I_ess_node) - P_A_ess_ch.dot(I_ess_node)
            for i in PF_BAL_PA.ravel():
                sr.addConstr(i == 0, "power_flow")

            PF_BAL_PB = P_B_gen.dot(I_gen_node) + P_B_load.dot(I_loadcap_node) + P_B_edge.dot(I_edge_node) + P_B_ess_disch.dot(I_ess_node) - P_B_ess_ch.dot(I_ess_node)
            for i in PF_BAL_PB.ravel():
                sr.addConstr(i == 0, "power_flow")

            PF_BAL_PC = P_C_gen.dot(I_gen_node) + P_C_load.dot(I_loadcap_node) + P_C_edge.dot(I_edge_node) + P_C_ess_disch.dot(I_ess_node) - P_C_ess_ch.dot(I_ess_node)
            for i in PF_BAL_PC.ravel():
                sr.addConstr(i == 0, "power_flow")

            PF_BAL_QA = Q_A_gen.dot(I_gen_node) + Q_A_load.dot(I_loadcap_node) + Q_A_edge.dot(I_edge_node) + Q_A_ess_disch.dot(I_ess_node) - Q_A_ess_ch.dot(I_ess_node)
            for i in PF_BAL_QA.ravel():
                sr.addConstr(i == 0, "power_flow")

            PF_BAL_QB = Q_B_gen.dot(I_gen_node) + Q_B_load.dot(I_loadcap_node) + Q_B_edge.dot(I_edge_node) + Q_B_ess_disch.dot(I_ess_node) - Q_B_ess_ch.dot(I_ess_node)
            for i in PF_BAL_QB.ravel():
                sr.addConstr(i == 0, "power_flow")

            PF_BAL_QC = Q_C_gen.dot(I_gen_node) + Q_C_load.dot(I_loadcap_node) + Q_C_edge.dot(I_edge_node) + Q_C_ess_disch.dot(I_ess_node) - Q_C_ess_ch.dot(I_ess_node)
            for i in PF_BAL_QC.ravel():
                sr.addConstr(i == 0, "power_flow")

        # Power flow voltage constraints  Check out Bo Chen's MATLAB Code
        if st_pf_voltage == True:
            for t in range(rh_horizon):
                for e in range(n_edge):
                    Nfrom = edge_list[e].Number_A
                    Nto   = edge_list[e].Number_B
                    Phases= edge_list[e].Phases
                    if edge_list[e].Config != 'regulator':
                        ZZ = np.diag(Phases).dot(Z[(3 * Nfrom):(3 * (Nfrom+1)), (3 * Nto):(3 * (Nto+1))])
                        RR = np.real(ZZ)
                        XX = np.imag(ZZ)

                        a  = np.array([[1], [np.exp(1j*-120/180*np.pi)], [np.exp(1j*+120/180*np.pi)]])
                        Requal = np.real(a.dot(a.conj().T)) * RR + np.imag(a.dot(a.conj().T)) * XX
                        Xequal = np.real(a.dot(a.conj().T)) * XX + np.imag(a.dot(a.conj().T)) * RR

                        Vf2 = np.array([V2_A_node[t, Nfrom], V2_B_node[t, Nfrom], V2_C_node[t, Nfrom]])
                        Vt2 = np.array([V2_A_node[t, Nto],   V2_B_node[t, Nto],   V2_C_node[t, Nto]])

                        Pft = np.array([P_A_edge[t, e], P_B_edge[t, e], P_C_edge[t, e]])
                        Qft = np.array([Q_A_edge[t, e], Q_B_edge[t, e], Q_C_edge[t, e]])

                        # Put 3x1 vector on the left then + a 1x1 number. Reverse it will not work!
                        Left  =  -(Vt2 - Vf2 + 2 * (Requal.dot(Pft) + Xequal.dot(Qft))) - (sr_M * (1 - x_edge[t, e]))
                        Right =   (Vt2 - Vf2 + 2 * (Requal.dot(Pft) + Xequal.dot(Qft))) - (sr_M * (1 - x_edge[t, e]))

                        for c in Left:
                            sr.addConstr(c <= 0, "power_flow")
                        for c in Right:
                            sr.addConstr(c <= 0, "power_flow")



                    else:
                        for r in range(n_regulator):
                            if regulator_list[r].Node == edge_list[e].Node:
                                rg = r

                        if sr_rg_enable == False:
                            if edge_list[e].Phases[0] == 1:
                                sr.addConstr( -(V2_A_node[t, Nto] - edge_list[e].Position[0]**2 * V2_A_node[t, Nfrom]) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr(  (V2_A_node[t, Nto] - edge_list[e].Position[0]**2 * V2_A_node[t, Nfrom]) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                            if edge_list[e].Phases[1] == 1:
                                sr.addConstr( -(V2_B_node[t, Nto] - edge_list[e].Position[1]**2 * V2_B_node[t, Nfrom]) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr(  (V2_B_node[t, Nto] - edge_list[e].Position[1]**2 * V2_B_node[t, Nfrom]) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                            if edge_list[e].Phases[2] == 1:
                                sr.addConstr( -(V2_C_node[t, Nto] - edge_list[e].Position[2]**2 * V2_C_node[t, Nfrom]) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr(  (V2_C_node[t, Nto] - edge_list[e].Position[2]**2 * V2_C_node[t, Nfrom]) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                        else:
                            if edge_list[e].Phases[0] == 1:
                                sr.addConstr( -(V2_A_node[t, Nto] - (VG_A_ratio2[t, rg]+1.01*V2_A_node[t, Nfrom]-1.01)) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr(  (V2_A_node[t, Nto] - (VG_A_ratio2[t, rg]+1.01*V2_A_node[t, Nfrom]-1.01)) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr( -(VG_A_ratio2[t, rg] - edge_list[e].Position[0] ** 2) - sr_M * x_edge[t, e] <= 0, "power_flow")
                                sr.addConstr(  (VG_A_ratio2[t, rg] - edge_list[e].Position[0] ** 2) - sr_M * x_edge[t, e] <= 0, "power_flow")
                                sr.addConstr(VG_A_ratio2[t, rg] >= 0.81, "regulator")
                                sr.addConstr(VG_A_ratio2[t, rg] <= 1.21, "regulator")
                            if edge_list[e].Phases[1] == 1:
                                sr.addConstr( -(V2_B_node[t, Nto] - (VG_B_ratio2[t, rg]+1.01*V2_B_node[t, Nfrom]-1.01)) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr(  (V2_B_node[t, Nto] - (VG_B_ratio2[t, rg]+1.01*V2_B_node[t, Nfrom]-1.01)) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr( -(VG_B_ratio2[t, rg] - edge_list[e].Position[1] ** 2) - sr_M * x_edge[t, e] <= 0, "power_flow")
                                sr.addConstr(  (VG_B_ratio2[t, rg] - edge_list[e].Position[1] ** 2) - sr_M * x_edge[t, e] <= 0, "power_flow")
                                sr.addConstr(VG_B_ratio2[t, rg] >= 0.81, "regulator")
                                sr.addConstr(VG_B_ratio2[t, rg] <= 1.21, "regulator")
                            if edge_list[e].Phases[2] == 1:
                                sr.addConstr( -(V2_C_node[t, Nto] - (VG_C_ratio2[t, rg]+1.01*V2_C_node[t, Nfrom]-1.01)) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr(  (V2_C_node[t, Nto] - (VG_C_ratio2[t, rg]+1.01*V2_C_node[t, Nfrom]-1.01)) - (sr_M*(1-x_edge[t,e])) <= 0, "power_flow")
                                sr.addConstr( -(VG_C_ratio2[t, rg] - edge_list[e].Position[2] ** 2) - sr_M * x_edge[t, e] <= 0, "power_flow")
                                sr.addConstr(  (VG_C_ratio2[t, rg] - edge_list[e].Position[2] ** 2) - sr_M * x_edge[t, e] <= 0, "power_flow")
                                sr.addConstr(VG_C_ratio2[t, rg] >= 0.81, "regulator")
                                sr.addConstr(VG_C_ratio2[t, rg] <= 1.21, "regulator")

        # Voltage limit constraints
        if st_voltage_limit == True:
            for t in range(rh_horizon):
                # black start DG voltage
                for g in range(n_gen):
                    if gen_list[g].Control == True:  # if the generator is controllable, @@no controllable != outage@@
                        if gen_list[g].Outage == False:  # if the generator is damaged.
                            if gen_list[g].Type == 'black start':
                                if gen_list[g].Phases[0] == 1:
                                    sr.addConstr(V2_A_node[t, gen_list[g].Number] == x_node[t, gen_list[g].Number] * sr_Vsrc**2, "voltage")
                                if gen_list[g].Phases[1] == 1:
                                    sr.addConstr(V2_B_node[t, gen_list[g].Number] == x_node[t, gen_list[g].Number] * sr_Vsrc**2, "voltage")
                                if gen_list[g].Phases[2] == 1:
                                    sr.addConstr(V2_C_node[t, gen_list[g].Number] == x_node[t, gen_list[g].Number] * sr_Vsrc**2, "voltage")
                # voltage limit
                for n in range(n_node):
                    sr.addConstr(V2_A_node[t, n] >= 0.95 * 0.95 * x_node[t, n], "voltage")
                    sr.addConstr(V2_A_node[t, n] <= 1.05 * 1.05 * x_node[t, n], "voltage")
                    sr.addConstr(V2_B_node[t, n] >= 0.95 * 0.95 * x_node[t, n], "voltage")
                    sr.addConstr(V2_B_node[t, n] <= 1.05 * 1.05 * x_node[t, n], "voltage")
                    sr.addConstr(V2_C_node[t, n] >= 0.95 * 0.95 * x_node[t, n], "voltage")
                    sr.addConstr(V2_C_node[t, n] <= 1.05 * 1.05 * x_node[t, n], "voltage")

        # Line capacity constraints
        if st_line_capacity == True:
            for t in range(rh_horizon):
                for e in range(n_edge):
                    if sr_n_polygon == 6:
                        plg = edge_list[e].Ampacity * np.sqrt(2 * np.pi / sr_n_polygon / np.sin(2 * np.pi / sr_n_polygon))
                        sr.addConstr(-1 * np.sqrt(3) * (P_A_edge[t, e] + plg) - Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     np.sqrt(3) * (P_A_edge[t, e] - plg) + Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) * (P_B_edge[t, e] + plg) - Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     np.sqrt(3) * (P_B_edge[t, e] - plg) + Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) * (P_C_edge[t, e] + plg) - Q_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     np.sqrt(3) * (P_C_edge[t, e] - plg) + Q_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) / 2 * plg - Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) / 2 * plg + Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) / 2 * plg - Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) / 2 * plg + Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) / 2 * plg - Q_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-1 * np.sqrt(3) / 2 * plg + Q_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr( np.sqrt(3) * (P_A_edge[t, e] - plg) - Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-np.sqrt(3) * (P_A_edge[t, e] + plg) + Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr( np.sqrt(3) * (P_B_edge[t, e] - plg) - Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-np.sqrt(3) * (P_B_edge[t, e] + plg) + Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr( np.sqrt(3) * (P_C_edge[t, e] - plg) - Q_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(-np.sqrt(3) * (P_C_edge[t, e] + plg) + Q_C_edge[t, e] <= 0, "line_capacity")

                    if sr_n_polygon == 4:
                        plg = edge_list[e].Ampacity * ( sr_Vbase / np.sqrt(3) ) / sr_Sbase / np.sqrt(2)
                        sr.addConstr(-1 * plg - P_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     plg - P_A_edge[t, e] >= 0, "line_capacity")
                        sr.addConstr(-1 * plg - P_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     plg - P_B_edge[t, e] >= 0, "line_capacity")
                        sr.addConstr(-1 * plg - P_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     plg - P_C_edge[t, e] >= 0, "line_capacity")
                        sr.addConstr(-1 * plg - Q_A_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     plg - Q_A_edge[t, e] >= 0, "line_capacity")
                        sr.addConstr(-1 * plg - Q_B_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     plg - Q_B_edge[t, e] >= 0, "line_capacity")
                        sr.addConstr(-1 * plg - Q_C_edge[t, e] <= 0, "line_capacity")
                        sr.addConstr(     plg - Q_C_edge[t, e] >= 0, "line_capacity")

        # Generator capacity constraints
        if st_gen_capacity == True:
            for t in range(1, rh_horizon):
                for g in range(n_gen):
                    sr.addConstr( 1 * gen_list[g].Pmin * x_gen[t, g] - (P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]) <= 0, "gen_capacity")
                    sr.addConstr(-1 * gen_list[g].Pmax * x_gen[t, g] + (P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]) <= 0, "gen_capacity")
                    sr.addConstr( 1 * gen_list[g].Qmin * x_gen[t, g] - (Q_A_gen[t, g] + Q_B_gen[t, g] + Q_C_gen[t, g]) <= 0, "gen_capacity")
                    sr.addConstr(-1 * gen_list[g].Qmax * x_gen[t, g] + (Q_A_gen[t, g] + Q_B_gen[t, g] + Q_C_gen[t, g]) <= 0, "gen_capacity")

        # Generator reserve constraints
        if st_gen_reserve == True:
            for t in range(rh_horizon):
                for g in range(n_gen):
                    sr.addConstr((P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]) - (1 - sr_reserve_margin) * gen_list[g].Pcap * x_gen[t, g] <= 0, "gen_reserve")

        # Generator Ramping Rate constraints
        if st_gen_ramp == True:
            for t in range(1, rh_horizon):
                for g in range(n_gen):
                    sr.addConstr((P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]) - (P_A_gen[t - 1, g] + P_B_gen[t - 1, g] + P_C_gen[t - 1, g]) - gen_list[g].Rmax * rh_step_length <= 0, "gen_ramp" )
                    sr.addConstr((P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]) - (P_A_gen[t - 1, g] + P_B_gen[t - 1, g] + P_C_gen[t - 1, g]) + gen_list[g].Rmax * rh_step_length >= 0, "gen_ramp")

        # Generator StepLoad constraints
        if st_gen_stepload == True:
            for t in range(1, rh_horizon):
                for g in range(n_gen):
                    sr.addConstr((P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]) - (P_A_gen[t - 1, g] + P_B_gen[t - 1, g] + P_C_gen[t - 1, g]) - gen_list[g].FRR * gen_list[g].Pcap <= 0, "gen_stepload")

        # Generator Unbalance Condition constraints:
        if st_gen_unbalance == True:
            for t in range(rh_horizon):
                for g in range(n_gen):
                    if gen_list[g].Type == "black start":
                        Pn = P_A_gen[t, g] + np.sqrt(3) / 2 * Q_B_gen[t, g] - 0.5 * P_B_gen[t, g] - np.sqrt(3) / 2 * Q_C_gen[t, g] - 0.5 * P_C_gen[t, g]
                        Qn = Q_A_gen[t, g] + np.sqrt(3) / 2 * P_C_gen[t, g] - 0.5 * Q_B_gen[t, g] - np.sqrt(3) / 2 * P_B_gen[t, g] - 0.5 * Q_C_gen[t, g]
                        Pp = P_A_gen[t, g] + P_B_gen[t, g] + P_C_gen[t, g]
                        Qp = Q_A_gen[t, g] + Q_B_gen[t, g] + Q_C_gen[t, g]

                        sr.addConstr(ypn[t, g] - Pn >= 0,                 "gen_unbalance")
                        sr.addConstr(ypn[t, g] - Pn <= sr_M * dpn1[t, g], "gen_unbalance")
                        sr.addConstr(ypn[t, g] + Pn >= 0,                 "gen_unbalance")
                        sr.addConstr(ypn[t, g] + Pn <= sr_M * dpn2[t, g], "gen_unbalance")
                        sr.addConstr(dpn1[t, g] + dpn2[t, g] == 1,        "gen_unbalance")

                        sr.addConstr(yqn[t, g] - Qn >= 0,                 "gen_unbalance")
                        sr.addConstr(yqn[t, g] - Qn <= sr_M * dqn1[t, g], "gen_unbalance")
                        sr.addConstr(yqn[t, g] + Qn >= 0,                 "gen_unbalance")
                        sr.addConstr(yqn[t, g] + Qn <= sr_M * dqn2[t, g], "gen_unbalance")
                        sr.addConstr(dqn1[t, g] + dqn2[t, g] == 1,        "gen_unbalance")

                        sr.addConstr(yqp[t, g] - Qp >= 0,                 "gen_unbalance")
                        sr.addConstr(yqp[t, g] - Qp <= sr_M * dqp1[t, g], "gen_unbalance")
                        sr.addConstr(yqp[t, g] + Qp >= 0,                 "gen_unbalance")
                        sr.addConstr(yqp[t, g] + Qp <= sr_M * dqp2[t, g], "gen_unbalance")
                        sr.addConstr(dqp1[t, g] + dqp2[t, g] == 1,        "gen_unbalance")

                        sr.addConstr(ypn[t, g] - ynmax[t, g] <= 0,        "gen_unbalance")
                        sr.addConstr(yqn[t, g] - ynmax[t, g] <= 0,        "gen_unbalance")
                        sr.addConstr(ypn[t, g] + sr_M * (1 - dnmax1[t, g]) - ynmax[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(yqn[t, g] + sr_M * (1 - dnmax2[t, g]) - ynmax[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(dnmax1[t, g] + dnmax2[t, g] == 1,    "gen_unbalance")

                        sr.addConstr(ypn[t, g] - ynmin[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(yqn[t, g] - ynmin[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(ypn[t, g] - sr_M * (1 - dnmin1[t, g]) - ynmin[t, g] <= 0, "gen_unbalance")
                        sr.addConstr(yqn[t, g] - sr_M * (1 - dnmin2[t, g]) - ynmin[t, g] <= 0, "gen_unbalance")
                        sr.addConstr(dnmin1[t, g] + dnmin2[t, g] == 1, "gen_unbalance")

                        sr.addConstr(ypp[t, g] == Pp, "gen_unbalance")
                        sr.addConstr(ypmax[t, g] - ypp[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(ypmax[t, g] - yqp[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(ypp[t, g] + sr_M * (1 - dpmax1[t, g]) - ypmax[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(yqn[t, g] + sr_M * (1 - dpmax2[t, g]) - ypmax[t, g] >= 0, "gen_unbalance")
                        sr.addConstr(dpmax1[t, g] + dpmax2[t, g] == 1, "gen_unbalance")
                        sr.addConstr(ypmin[t, g] - ypp[t, g] <= 0, "gen_unbalance")
                        sr.addConstr(ypmin[t, g] - yqp[t, g] <= 0, "gen_unbalance")
                        sr.addConstr(ypp[t, g] - sr_M * (1 - dpmin1[t, g]) - ypmin[t, g] <= 0, "gen_unbalance")
                        sr.addConstr(yqp[t, g] - sr_M * (1 - dpmin2[t, g]) - ypmin[t, g] <= 0, "gen_unbalance")
                        sr.addConstr(dpmin1[t, g] + dpmin2[t, g] == 1, "gen_unbalance")

                        sr.addConstr(0.9375 * ynmax[t, g] + 0.4688 * ynmin[t, g] - gen_list[g].CUI * (0.9375 * ypmax[t, g] + 0.4688 * ypmin[t, g]) <= 0, "gen_unbalance")

        ###################################################################################################
        # Non-shed constraints
        if st_non_shed == True:
            for i in range(1, rh_horizon):
                for j in range(n_gen):
                    sr.addConstr(x_gen[i,j] - x_gen[i-1, j] >= 0, "non_shed_gen")
                for j in range(n_edge):
                    sr.addConstr(x_edge[i,j] - x_edge[i-1, j] >= 0, "non_shed_edge")
                for j in range(n_load):
                    sr.addConstr(x_load[i,j] - x_load[i-1, j] >= 0, "non-shed_load")

        # connectivity constraints
        if st_connectivity == True:
            # generator
            for i in range(rh_horizon):
                for g in range(n_gen):
                    if gen_list[g].Control == True:  # if the generator is controllable, @@no controllable != outage@@
                        if gen_list[g].Outage == False:  # if the generator is damaged.
                            if gen_list[g].Type == 'non black start':
                                sr.addConstr(x_gen[i, g] <= x_node[i, gen_list[g].Number], "connectivity_gen")
                            elif gen_list[g].Type == 'black start':
                                # sr.addConstr(x_gen[i, g] == 1, "connectivity_gen")
                                sr.addConstr(x_gen[i, g] <= x_node[i, gen_list[g].Number], "connectivity_gen")
                        else:
                            sr.addConstr(x_gen[i, g] == 0, "connectivity_gen")
                    else:
                        # sr.addConstr(x_gen[i, g] == x_node[i, gen_list[g].Number], "connectivity_gen")
                        sr.addConstr(x_gen[i, g] == 0, "connectivity_gen")

            # edge
            for i in range(rh_horizon):
                for e in range(n_edge):
                    if edge_list[e].Outage == False:
                        if edge_list[e].Control == True:
                            sr.addConstr(x_edge[i, e] <= x_node[i, edge_list[e].Number_A], "connectivity_edge")
                            sr.addConstr(x_edge[i, e] <= x_node[i, edge_list[e].Number_B], "connectivity_edge")
                            if i > 0:
                                sr.addConstr(x_edge[i, e] - x_edge[i-1, e] <=
                                             (x_node[i, edge_list[e].Number_A] - x_node[i-1, edge_list[e].Number_A])
                                             + (x_node[i, edge_list[e].Number_B] - x_node[i-1, edge_list[e].Number_B]), "connectivity_edge")
                        else:
                            sr.addConstr(x_edge[i, e] == x_node[i, edge_list[e].Number_A], "connectivity_edge")
                            sr.addConstr(x_edge[i, e] == x_node[i, edge_list[e].Number_B], "connectivity_edge")
                    else:
                        sr.addConstr(x_edge[i,e] == 0, "connectivity_edge")

            # node
            for i in range(rh_horizon):
                for j in range(n_node):
                    if node_list[j].Outage == True:
                        sr.addConstr(x_node[i, j] == 0, "connectivity_node")
                for g in range(n_gen):
                    if gen_list[g].Control == True:  # if the generator is controllable, @@no controllable != outage@@
                        if gen_list[g].Outage == False:  # if the generator is damaged.
                            if gen_list[g].Type == 'black start':
                                # pass
                                sr.addConstr(x_gen[i, g] == x_node[i, gen_list[g].Number], "connectivity_node")

            # load
            for i in range(rh_horizon):
                for j in range(n_loadcap):
                    if loadcap_list[j].Outage == False:
                        if loadcap_list[j].Control == True:
                            sr.addConstr(x_load[i, j] <= x_node[i, loadcap_list[j].Number], "connectivity_load")
                        else:
                            sr.addConstr(x_load[i, j] == x_node[i, loadcap_list[j].Number], "connectivity_load")
                    else:
                        sr.addConstr(x_load[i, j] == 0, "connectivity_load")

        # Sequence constraints: see the paper published on Trans on Smart Grid
        if st_sequence == True:
            for i in range(1, rh_horizon):
                for j in range(n_edge):
                    if edge_list[j].Control == True:
                        sr.addConstr(x_edge[i,j] <= x_node[i-1, edge_list[j].Number_A] + x_node[i-1, edge_list[j].Number_B], "sequence_edge")

            # Bus Block Method based on connected components in a graph
            # find all the nodes connected to black start DGs
            bs_node = set()
            for i in gen_list:
                if i.Type == "black start":
                    if i.Control == True:
                        if i.Outage == False:
                            bs_node.update([i.Node])

            for block in islands:
                # constraints only apply for blocks without black start DGs
                if len(bs_node.intersection(block)) == 0:
                    # find all the switchable edges connected to a block
                    edge_indices = []
                    for a_node in block:
                        for b_node in edge_dict[a_node]:
                            if edge_dict[a_node][b_node]["Control"] == True:
                                edge_indices.append(edge_dict[a_node][b_node]["Index"])

                    # find a representative node in the block
                    node_index = node_dict[next(iter(block))]["Number"]

                    for t in range(rh_horizon):
                        conn_lines = 0
                        for j in edge_indices:
                            conn_lines += x_edge[t, j]
                        # Add constraints (33b) in TPWR paper
                        sr.addConstr(x_node[t, node_index] <= conn_lines, "sequence_edge")
                        # Add constriants (32c) in TPWR paper
                        if t > 0:
                            conn_lines_previous = 0
                            for j in edge_indices:
                                conn_lines_previous += x_edge[t-1, j]
                            sr.addConstr(conn_lines - conn_lines_previous <= 1 + sr_M * x_node[t-1, node_index], "sequence_edge")

        # Topological constraints: radial for each island
        if st_topology == True:
            for i in range(rh_horizon):
                on_gen = 0
                for g in range(n_gen):
                    if gen_list[g].Type == "black start":
                        on_gen += x_gen[i, g]
                on_node = 0
                for n in range(n_node):
                    on_node += x_node[i, n]
                on_edge = 0
                for e in range(n_edge):
                    on_edge += x_edge[i,e]
                sr.addConstr(on_gen == on_node - on_edge, "topology")

        # Initiation constraints
        if st_init:
            if iteration == 0:
                # set all the switchable lines as disconnected
                for j in range(n_edge):
                    if edge_list[j].Control == False:
                        pass
                    elif edge_list[j].Status == True:
                        sr.addConstr(x_edge[0, j] == 1, "init_edge")
                    else:
                        sr.addConstr(x_edge[0, j] == 0, "init_edge")
            else:
                start_step = (rh_control - 1) * iteration
                for g in range(n_gen):
                    sr.addConstr(x_gen[0, g] == sltn_x_gen[start_step, g])
                    sr.addConstr(P_A_gen[0, g] == sltn_P_A_gen[start_step, g])
                    sr.addConstr(P_B_gen[0, g] == sltn_P_B_gen[start_step, g])
                    sr.addConstr(P_C_gen[0, g] == sltn_P_C_gen[start_step, g])
                    sr.addConstr(Q_A_gen[0, g] == sltn_Q_A_gen[start_step, g])
                    sr.addConstr(Q_B_gen[0, g] == sltn_Q_B_gen[start_step, g])
                    sr.addConstr(Q_C_gen[0, g] == sltn_Q_C_gen[start_step, g])
                for e in range(n_edge):
                    sr.addConstr(x_edge[0, e] == sltn_x_edge[start_step, e])
                for l in range(n_loadcap):
                    sr.addConstr(x_load[0, l] == sltn_x_loadcap[start_step, l])
                    sr.addConstr(P_A_load[0, l] == sltn_P_A_load[start_step, l])
                    sr.addConstr(P_B_load[0, l] == sltn_P_B_load[start_step, l])
                    sr.addConstr(P_C_load[0, l] == sltn_P_C_load[start_step, l])
                    sr.addConstr(Q_A_load[0, l] == sltn_Q_A_load[start_step, l])
                    sr.addConstr(Q_B_load[0, l] == sltn_Q_B_load[start_step, l])
                    sr.addConstr(Q_C_load[0, l] == sltn_Q_C_load[start_step, l])
                for n in range(n_node):
                    sr.addConstr(x_node[0, n] == sltn_x_node[start_step, n])
                for e in range(n_ess):
                    sr.addConstr(x_ess_ch[0, e] == sltn_x_ess_ch[start_step, e])
                    sr.addConstr(x_ess_disch[0, e] == sltn_x_ess_disch[start_step, e])
                    sr.addConstr(P_A_ess_ch[0, e] == sltn_P_A_ess_ch[start_step, e])
                    sr.addConstr(P_B_ess_ch[0, e] == sltn_P_B_ess_ch[start_step, e])
                    sr.addConstr(P_C_ess_ch[0, e] == sltn_P_C_ess_ch[start_step, e])
                    sr.addConstr(Q_A_ess_ch[0, e] == sltn_Q_A_ess_ch[start_step, e])
                    sr.addConstr(Q_B_ess_ch[0, e] == sltn_Q_B_ess_ch[start_step, e])
                    sr.addConstr(Q_C_ess_ch[0, e] == sltn_Q_C_ess_ch[start_step, e])
                    sr.addConstr(P_A_ess_disch[0, e] == sltn_P_A_ess_disch[start_step, e])
                    sr.addConstr(P_B_ess_disch[0, e] == sltn_P_B_ess_disch[start_step, e])
                    sr.addConstr(P_C_ess_disch[0, e] == sltn_P_C_ess_disch[start_step, e])
                    sr.addConstr(Q_A_ess_disch[0, e] == sltn_Q_A_ess_disch[start_step, e])
                    sr.addConstr(Q_B_ess_disch[0, e] == sltn_Q_B_ess_disch[start_step, e])
                    sr.addConstr(Q_C_ess_disch[0, e] == sltn_Q_C_ess_disch[start_step, e])
                # Update CLPU profile after each iteration
                if sr_clpu_enable == True:
                    for l in range(n_loadcap):
                        last_start = (rh_control - 1) * (iteration - 1)
                        on_steps = int((np.sum(sltn_x_loadcap[last_start:last_start+(rh_control), l])))
                        if on_steps > 0:
                            if on_steps <= len(loadcap_list[l].CLPU_profile):
                                CL_accumu = np.sum(loadcap_list[l].CLPU_profile[0:on_steps+1])
                                loadcap_list[l].CLPU_profile = np.concatenate((loadcap_list[l].CLPU_profile[on_steps:], np.zeros(on_steps+1)))
                                loadcap_list[l].CLPU_profile[0] = CL_accumu
                            else: # rarely happen
                                CL_accumu = np.sum(loadcap_list[l].CLPU_profile)
                                loadcap_list[l].CLPU_profile = np.concatenate((np.array([CL_accumu]), np.zeros(100)))

        ###################################################################################################
        # Load Model:
        if sr_clpu_enable == True:
            # this code is for single-iteration. The updaing policy for multiple iterations is different
            if iteration == 0:
                for l in loadcap_list:
                    l.clpu_generation(rh_step_length)

            for t in range(rh_horizon):
                if iteration > 0 and t == 0:
                    continue
                else:
                    for l in range(n_loadcap):
                        CLPU_overshot = loadcap_list[l].CLPU_factor[0]
                        C_accumu = 0
                        for k in range(t+1):
                            C_accumu += loadcap_list[l].CLPU_profile[k] * x_load[t-k, l]  # Double Check
                        # applicable for load and cap?
                        sr.addConstr(P_A_load[t, l] == loadcap_list[l].P[0] * CLPU_overshot * x_load[t, l] + loadcap_list[l].P[0] * C_accumu, "CLPU")
                        sr.addConstr(P_B_load[t, l] == loadcap_list[l].P[1] * CLPU_overshot * x_load[t, l] + loadcap_list[l].P[1] * C_accumu, "CLPU")
                        sr.addConstr(P_C_load[t, l] == loadcap_list[l].P[2] * CLPU_overshot * x_load[t, l] + loadcap_list[l].P[2] * C_accumu, "CLPU")
                        sr.addConstr(Q_A_load[t, l] == loadcap_list[l].Q[0] * CLPU_overshot * x_load[t, l] + loadcap_list[l].Q[0] * C_accumu, "CLPU")
                        sr.addConstr(Q_B_load[t, l] == loadcap_list[l].Q[1] * CLPU_overshot * x_load[t, l] + loadcap_list[l].Q[1] * C_accumu, "CLPU")
                        sr.addConstr(Q_C_load[t, l] == loadcap_list[l].Q[2] * CLPU_overshot * x_load[t, l] + loadcap_list[l].Q[2] * C_accumu, "CLPU")
        else:
            for t in range(rh_horizon):
                for l in range(n_loadcap):
                    sr.addConstr(P_A_load[t, l] - x_load[t, l] * loadcap_list[l].P[0] == 0, "load_model")
                    sr.addConstr(P_B_load[t, l] - x_load[t, l] * loadcap_list[l].P[1] == 0, "load_model")
                    sr.addConstr(P_C_load[t, l] - x_load[t, l] * loadcap_list[l].P[2] == 0, "load_model")
                    sr.addConstr(Q_A_load[t, l] - x_load[t, l] * loadcap_list[l].Q[0] == 0, "load_model")
                    sr.addConstr(Q_B_load[t, l] - x_load[t, l] * loadcap_list[l].Q[1] == 0, "load_model")
                    sr.addConstr(Q_C_load[t, l] - x_load[t, l] * loadcap_list[l].Q[2] == 0, "load_model")

        # ESS Model
        if sr_es_enable == True:
            for t in range(rh_horizon):
                # energization status
                for e in range(n_ess):
                    sr.addConstr(0 <= P_A_ess_ch[t, e], "power_flow")
                    sr.addConstr(0 <= P_B_ess_ch[t, e], "power_flow")
                    sr.addConstr(0 <= P_C_ess_ch[t, e], "power_flow")
                    sr.addConstr(0 <= P_A_ess_disch[t, e], "power_flow")
                    sr.addConstr(0 <= P_B_ess_disch[t, e], "power_flow")
                    sr.addConstr(0 <= P_C_ess_disch[t, e], "power_flow")
                    sr.addConstr(P_A_ess_ch[t, e] - sr_M * x_ess_ch[t, e] <= 0, "power_flow")
                    sr.addConstr(P_B_ess_ch[t, e] - sr_M * x_ess_ch[t, e] <= 0, "power_flow")
                    sr.addConstr(P_C_ess_ch[t, e] - sr_M * x_ess_ch[t, e] <= 0, "power_flow")
                    sr.addConstr(P_A_ess_disch[t, e] - sr_M * x_ess_disch[t, e] <= 0, "power_flow")
                    sr.addConstr(P_B_ess_disch[t, e] - sr_M * x_ess_disch[t, e] <= 0, "power_flow")
                    sr.addConstr(P_C_ess_disch[t, e] - sr_M * x_ess_disch[t, e] <= 0, "power_flow")
                    if ess_list[e].Phases[0] == 0:
                        sr.addConstr(0 == P_A_ess_disch[t, e], "power_flow")
                        sr.addConstr(0 == P_A_ess_ch[t, e], "power_flow")
                        sr.addConstr(0 == Q_A_ess_disch[t, e], "power_flow")
                        sr.addConstr(0 == Q_A_ess_ch[t, e], "power_flow")
                    if ess_list[e].Phases[1] == 0:
                        sr.addConstr(0 == P_B_ess_disch[t, e], "power_flow")
                        sr.addConstr(0 == P_B_ess_ch[t, e], "power_flow")
                        sr.addConstr(0 == Q_B_ess_disch[t, e], "power_flow")
                        sr.addConstr(0 == Q_B_ess_ch[t, e], "power_flow")
                    if ess_list[e].Phases[2] == 0:
                        sr.addConstr(0 == P_C_ess_disch[t, e], "power_flow")
                        sr.addConstr(0 == P_C_ess_ch[t, e], "power_flow")
                        sr.addConstr(0 == Q_C_ess_disch[t, e], "power_flow")
                        sr.addConstr(0 == Q_C_ess_ch[t, e], "power_flow")
                # operational limits
                for e in range(n_ess):
                    sr.addConstr(x_ess_ch[t, e] + x_ess_disch[t, e] <= x_node[t, ess_list[e].Number], "ess")
                    if ess_list[e].Phases[0] == 1:
                        sr.addConstr(ess_list[e].P_ch_min * x_ess_ch[t, e] - P_A_ess_ch[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].P_ch_max * x_ess_ch[t, e] + P_A_ess_ch[t, e] <= 0, "ess")
                        sr.addConstr(ess_list[e].P_disch_min * x_ess_disch[t, e] - P_A_ess_disch[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].P_disch_max * x_ess_disch[t, e] + P_A_ess_disch[t, e] <= 0, "ess")
                        sr.addConstr(ess_list[e].SOC_min - SOC_A[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].SOC_max + SOC_A[t, e] <= 0, "ess")
                    if ess_list[e].Phases[1] == 1:
                        sr.addConstr(ess_list[e].P_ch_min * x_ess_ch[t, e] - P_B_ess_ch[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].P_ch_max * x_ess_ch[t, e] + P_B_ess_ch[t, e] <= 0, "ess")
                        sr.addConstr(ess_list[e].P_disch_min * x_ess_disch[t, e] - P_B_ess_disch[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].P_disch_max * x_ess_disch[t, e] + P_B_ess_disch[t, e] <= 0, "ess")
                        sr.addConstr(ess_list[e].SOC_min - SOC_B[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].SOC_max + SOC_B[t, e] <= 0, "ess")
                    if ess_list[e].Phases[2] == 1:
                        sr.addConstr(ess_list[e].P_ch_min * x_ess_ch[t, e] - P_C_ess_ch[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].P_ch_max * x_ess_ch[t, e] + P_C_ess_ch[t, e] <= 0, "ess")
                        sr.addConstr(ess_list[e].P_disch_min * x_ess_disch[t, e] - P_C_ess_disch[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].P_disch_max * x_ess_disch[t, e] + P_C_ess_disch[t, e] <= 0, "ess")
                        sr.addConstr(ess_list[e].SOC_min - SOC_C[t, e] <= 0, "ess")
                        sr.addConstr(-ess_list[e].SOC_max + SOC_C[t, e] <= 0, "ess")
                # SOC ========
                for e in range(n_ess):
                    if t == 0:
                        if iteration == 0:
                            if ess_list[e].Phases[0] == 1:
                                sr.addConstr(SOC_A[t, e] == ess_list[e].SOC_init)
                            if ess_list[e].Phases[1] == 1:
                                sr.addConstr(SOC_B[t, e] == ess_list[e].SOC_init)
                            if ess_list[e].Phases[2] == 1:
                                sr.addConstr(SOC_C[t, e] == ess_list[e].SOC_init)
                        else:
                            start_step = (rh_control - 1) * iteration
                            if ess_list[e].Phases[0] == 1:
                                sr.addConstr(SOC_A[t, e] == sltn_SOC_A[start_step, e])
                            if ess_list[e].Phases[1] == 1:
                                sr.addConstr(SOC_B[t, e] == sltn_SOC_B[start_step, e])
                            if ess_list[e].Phases[2] == 1:
                                sr.addConstr(SOC_C[t, e] == sltn_SOC_C[start_step, e])
                    elif t > 0:
                        if ess_list[e].Phases[0] == 1:
                            sr.addConstr(SOC_A[t, e] - SOC_A[t-1, e]
                                         - ess_list[e].effi_ch * P_A_ess_ch[t, e]*rh_step_length/60
                                         + (1/ess_list[e].effi_disch) * P_A_ess_disch[t, e]*rh_step_length/60
                                         == 0, "ess")
                        if ess_list[e].Phases[1] == 1:
                            sr.addConstr(SOC_B[t, e] - SOC_B[t-1, e]
                                         - ess_list[e].effi_ch * P_B_ess_ch[t, e]*rh_step_length/60
                                         + (1/ess_list[e].effi_disch) * P_B_ess_disch[t, e]*rh_step_length/60
                                         == 0, "ess")
                        if ess_list[e].Phases[2] == 1:
                            sr.addConstr(SOC_C[t, e] - SOC_C[t-1, e]
                                         - ess_list[e].effi_ch * P_C_ess_ch[t, e]*rh_step_length/60
                                         + (1/ess_list[e].effi_disch) * P_C_ess_disch[t, e]*rh_step_length/60
                                         == 0, "ess")
        else:
            for t in range(rh_horizon):
                for e in range(n_ess):
                    sr.addConstr(P_A_ess_ch[t, e] == 0)
                    sr.addConstr(P_B_ess_ch[t, e] == 0)
                    sr.addConstr(P_C_ess_ch[t, e] == 0)
                    sr.addConstr(P_A_ess_disch[t, e] == 0)
                    sr.addConstr(P_B_ess_disch[t, e] == 0)
                    sr.addConstr(P_C_ess_disch[t, e] == 0)
                    sr.addConstr(Q_A_ess_ch[t, e] == 0)
                    sr.addConstr(Q_B_ess_ch[t, e] == 0)
                    sr.addConstr(Q_C_ess_ch[t, e] == 0)
                    sr.addConstr(Q_A_ess_disch[t, e] == 0)
                    sr.addConstr(Q_B_ess_disch[t, e] == 0)
                    sr.addConstr(Q_C_ess_disch[t, e] == 0)

        # Capacitor Model
        if sr_cap_enable == False:
            for i in range(rh_horizon):
                for c in range(n_loadcap):
                    if 'cap' in loadcap_list[c].Name:
                        status = int(loadcap_list[c].Status)
                        sr.addConstr(x_load[i, c] == status * x_node[i, loadcap_list[c].Number], "init_cap")

        ###################################################################################################
        sr.update()

        total_load = 0
        for i in range(rh_horizon):
            for j in range(n_loadcap):
                total_load += x_load[i, j]
        sr.setObjective(total_load, GRB.MAXIMIZE)

        # total_edge = 0
        # for i in range(rh_horizon):
        #     for j in range(n_edge):
        #         total_edge += x_edge[i, j]
        # sr.setObjective(total_edge, GRB.MAXIMIZE)

        sr.optimize()

        ####################################################################################################
        # print, organize, and return solutions
        for v in sr.getVars():
            a = v.varName
            key1 = a.find('[')
            key2 = a.find(',')
            key3 = a.find(']')
            key = a[:key2+1]
            number = int(a[key2+1:key3])
            if v.x != 0:
                if 'load' in key:
                    name = load_list[number].Node
                    print(key + name + ']=' '%g' % (v.x))
                if 'gen' in key:
                    name = gen_list[number].Node
                    print(key + name + ']=' '%g' % (v.x))
                if 'node' in key:
                    name = node_list[number].Node
                    print(key + name + ']=' '%g' % (v.x))
                if 'edge' in key:
                    name = edge_list[number].Name
                    print(key + name + ']=' '%g' % (v.x))
                if 'VG' in key:
                    name = regulator_list[number].Name
                    print(key + name + ']=' '%g' % (v.x))
                if 'ess' in key:
                    name = ess_list[number].Name
                    print(key + name + ']=' '%g' % (v.x))
                if 'SOC' in key:
                    name = ess_list[number].Name
                    print(key + name + ']=' '%g' % (v.x))

        ####################################################################################################
        if iteration == 0:
            sltn_x_gen     = np.zeros((rh_total_step, n_gen))
            sltn_P_A_gen   = np.zeros((rh_total_step, n_gen))
            sltn_P_B_gen   = np.zeros((rh_total_step, n_gen))
            sltn_P_C_gen   = np.zeros((rh_total_step, n_gen))
            sltn_Q_A_gen   = np.zeros((rh_total_step, n_gen))
            sltn_Q_B_gen   = np.zeros((rh_total_step, n_gen))
            sltn_Q_C_gen   = np.zeros((rh_total_step, n_gen))

            sltn_x_edge    = np.zeros((rh_total_step, n_edge))
            sltn_x_node    = np.zeros((rh_total_step, n_node))

            sltn_x_loadcap = np.zeros((rh_total_step, n_loadcap))
            sltn_P_A_load  = np.zeros((rh_total_step, n_loadcap))
            sltn_P_B_load  = np.zeros((rh_total_step, n_loadcap))
            sltn_P_C_load  = np.zeros((rh_total_step, n_loadcap))
            sltn_Q_A_load  = np.zeros((rh_total_step, n_loadcap))
            sltn_Q_B_load  = np.zeros((rh_total_step, n_loadcap))
            sltn_Q_C_load  = np.zeros((rh_total_step, n_loadcap))

            sltn_x_ess_ch    = np.zeros((rh_total_step, n_ess))
            sltn_x_ess_disch = np.zeros((rh_total_step, n_ess))
            sltn_P_A_ess_ch  = np.zeros((rh_total_step, n_ess))
            sltn_P_B_ess_ch = np.zeros((rh_total_step, n_ess))
            sltn_P_C_ess_ch = np.zeros((rh_total_step, n_ess))
            sltn_P_A_ess_disch  = np.zeros((rh_total_step, n_ess))
            sltn_P_B_ess_disch = np.zeros((rh_total_step, n_ess))
            sltn_P_C_ess_disch = np.zeros((rh_total_step, n_ess))
            sltn_Q_A_ess_ch  = np.zeros((rh_total_step, n_ess))
            sltn_Q_B_ess_ch = np.zeros((rh_total_step, n_ess))
            sltn_Q_C_ess_ch = np.zeros((rh_total_step, n_ess))
            sltn_Q_A_ess_disch  = np.zeros((rh_total_step, n_ess))
            sltn_Q_B_ess_disch = np.zeros((rh_total_step, n_ess))
            sltn_Q_C_ess_disch = np.zeros((rh_total_step, n_ess))
            sltn_SOC_A = np.zeros((rh_total_step, n_ess))
            sltn_SOC_B = np.zeros((rh_total_step, n_ess))
            sltn_SOC_C = np.zeros((rh_total_step, n_ess))

            for t in range(rh_horizon):
                for g in range(n_gen):
                    sltn_x_gen[t, g]   = sr.getVarByName('x_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_P_A_gen[t, g] = sr.getVarByName('P_A_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_P_B_gen[t, g] = sr.getVarByName('P_B_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_P_C_gen[t, g] = sr.getVarByName('P_C_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_Q_A_gen[t, g] = sr.getVarByName('Q_A_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_Q_B_gen[t, g] = sr.getVarByName('Q_B_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_Q_C_gen[t, g] = sr.getVarByName('Q_C_gen[' + str(t) + ',' + str(g) + ']').x
                for e in range(n_edge):
                    sltn_x_edge[t, e] = sr.getVarByName('x_edge[' + str(t) + ',' + str(e) + ']').x
                for l in range(n_loadcap):
                    sltn_x_loadcap[t, l] = sr.getVarByName('x_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_P_A_load[t, l]  = sr.getVarByName('P_A_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_P_B_load[t, l]  = sr.getVarByName('P_B_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_P_C_load[t, l]  = sr.getVarByName('P_C_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_Q_A_load[t, l]  = sr.getVarByName('Q_A_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_Q_B_load[t, l]  = sr.getVarByName('Q_B_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_Q_C_load[t, l]  = sr.getVarByName('Q_C_load[' + str(t) + ',' + str(l) + ']').x
                for n in range(n_node):
                    sltn_x_node[t, n] = sr.getVarByName('x_node[' + str(t) + ',' + str(n) + ']').x
                for e in range(n_ess):
                    sltn_x_ess_ch[t, e] = sr.getVarByName('x_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_x_ess_disch[t, e] = sr.getVarByName('x_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_A_ess_ch[t, e] = sr.getVarByName('P_A_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_B_ess_ch[t, e] = sr.getVarByName('P_B_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_C_ess_ch[t, e] = sr.getVarByName('P_C_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_A_ess_disch[t, e] = sr.getVarByName('P_A_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_B_ess_disch[t, e] = sr.getVarByName('P_B_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_C_ess_disch[t, e] = sr.getVarByName('P_C_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_A_ess_ch[t, e] = sr.getVarByName('Q_A_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_B_ess_ch[t, e] = sr.getVarByName('Q_B_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_C_ess_ch[t, e] = sr.getVarByName('Q_C_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_A_ess_disch[t, e] = sr.getVarByName('Q_A_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_B_ess_disch[t, e] = sr.getVarByName('Q_B_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_C_ess_disch[t, e] = sr.getVarByName('Q_C_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_SOC_A[t, e] = sr.getVarByName('SOC_A[' + str(t) + ',' + str(e) + ']').x
                    sltn_SOC_B[t, e] = sr.getVarByName('SOC_B[' + str(t) + ',' + str(e) + ']').x
                    sltn_SOC_C[t, e] = sr.getVarByName('SOC_C[' + str(t) + ',' + str(e) + ']').x
        else:
            for t in range(rh_horizon): # using rh_horizon instead of rh_control may over-writing some elements, but it'll simplify coding.
                pos = (rh_control - 1) * iteration + t
                for g in range(n_gen):
                    sltn_x_gen[pos, g] = sr.getVarByName('x_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_P_A_gen[pos, g] = sr.getVarByName('P_A_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_P_B_gen[pos, g] = sr.getVarByName('P_B_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_P_C_gen[pos, g] = sr.getVarByName('P_C_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_Q_A_gen[pos, g] = sr.getVarByName('Q_A_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_Q_B_gen[pos, g] = sr.getVarByName('Q_B_gen[' + str(t) + ',' + str(g) + ']').x
                    sltn_Q_C_gen[pos, g] = sr.getVarByName('Q_C_gen[' + str(t) + ',' + str(g) + ']').x
                for e in range(n_edge):
                    sltn_x_edge[pos, e] = sr.getVarByName('x_edge[' + str(t) + ',' + str(e) + ']').x
                for l in range(n_loadcap):
                    sltn_x_loadcap[pos, l] = sr.getVarByName('x_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_P_A_load[pos, l]  = sr.getVarByName('P_A_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_P_B_load[pos, l]  = sr.getVarByName('P_B_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_P_C_load[pos, l]  = sr.getVarByName('P_C_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_Q_A_load[pos, l]  = sr.getVarByName('Q_A_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_Q_B_load[pos, l]  = sr.getVarByName('Q_B_load[' + str(t) + ',' + str(l) + ']').x
                    sltn_Q_C_load[pos, l]  = sr.getVarByName('Q_C_load[' + str(t) + ',' + str(l) + ']').x
                for n in range(n_node):
                    sltn_x_node[pos, n] = sr.getVarByName('x_node[' + str(t) + ',' + str(n) + ']').x
                for e in range(n_ess):
                    sltn_x_ess_ch[pos, e] = sr.getVarByName('x_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_x_ess_disch[pos, e] = sr.getVarByName('x_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_A_ess_ch[pos, e] = sr.getVarByName('P_A_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_B_ess_ch[pos, e] = sr.getVarByName('P_B_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_C_ess_ch[pos, e] = sr.getVarByName('P_C_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_A_ess_disch[pos, e] = sr.getVarByName('P_A_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_B_ess_disch[pos, e] = sr.getVarByName('P_B_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_P_C_ess_disch[pos, e] = sr.getVarByName('P_C_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_A_ess_ch[pos, e] = sr.getVarByName('Q_A_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_B_ess_ch[pos, e] = sr.getVarByName('Q_B_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_C_ess_ch[pos, e] = sr.getVarByName('Q_C_ess_ch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_A_ess_disch[pos, e] = sr.getVarByName('Q_A_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_B_ess_disch[pos, e] = sr.getVarByName('Q_B_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_Q_C_ess_disch[pos, e] = sr.getVarByName('Q_C_ess_disch[' + str(t) + ',' + str(e) + ']').x
                    sltn_SOC_A[pos, e] = sr.getVarByName('SOC_A[' + str(t) + ',' + str(e) + ']').x
                    sltn_SOC_B[pos, e] = sr.getVarByName('SOC_B[' + str(t) + ',' + str(e) + ']').x
                    sltn_SOC_C[pos, e] = sr.getVarByName('SOC_C[' + str(t) + ',' + str(e) + ']').x
    ######################################################################################
    ######################################################################################

    # Send variable values to the plotting function
    sr_solution = solution()
    # sr_solution.objective = sr.getObjective().getValue()
    sr_solution.x_gen   = sltn_x_gen
    sr_solution.P_A_gen = sltn_P_A_gen
    sr_solution.P_B_gen = sltn_P_B_gen
    sr_solution.P_C_gen = sltn_P_C_gen
    sr_solution.Q_A_gen = sltn_Q_A_gen
    sr_solution.Q_B_gen = sltn_Q_B_gen
    sr_solution.Q_C_gen = sltn_Q_C_gen

    sr_solution.x_edge    = sltn_x_edge

    sr_solution.x_loadcap = sltn_x_loadcap
    sr_solution.P_A_load  = sltn_P_A_load
    sr_solution.P_B_load  = sltn_P_B_load
    sr_solution.P_C_load  = sltn_P_C_load
    sr_solution.Q_A_load  = sltn_Q_A_load
    sr_solution.Q_B_load  = sltn_Q_B_load
    sr_solution.Q_C_load  = sltn_Q_C_load

    sr_solution.x_node    = sltn_x_node

    sr_solution.x_ess_ch  = sltn_x_ess_ch
    sr_solution.x_ess_disch = sltn_x_ess_disch
    sr_solution.P_A_ess_ch = sltn_P_A_ess_ch
    sr_solution.P_B_ess_ch = sltn_P_B_ess_ch
    sr_solution.P_C_ess_ch = sltn_P_C_ess_ch
    sr_solution.Q_A_ess_ch = sltn_Q_A_ess_ch
    sr_solution.Q_B_ess_ch = sltn_Q_B_ess_ch
    sr_solution.Q_C_ess_ch = sltn_Q_C_ess_ch
    sr_solution.P_A_ess_disch = sltn_P_A_ess_disch
    sr_solution.P_B_ess_disch = sltn_P_B_ess_disch
    sr_solution.P_C_ess_disch = sltn_P_C_ess_disch
    sr_solution.Q_A_ess_disch = sltn_Q_A_ess_disch
    sr_solution.Q_B_ess_disch = sltn_Q_B_ess_disch
    sr_solution.Q_C_ess_disch = sltn_Q_C_ess_disch
    sr_solution.SOC_A = sltn_SOC_A
    sr_solution.SOC_B = sltn_SOC_B
    sr_solution.SOC_C = sltn_SOC_C
    sr_solution.rh_setup    = rh_setup
    sr_solution.sr_setup    = sr_setup

    obj = [sr_solution.x_gen, sr_solution.P_A_gen, sr_solution.P_B_gen, sr_solution.P_C_gen, sr_solution.Q_A_gen, sr_solution.Q_B_gen, sr_solution.Q_C_gen,
           sr_solution.x_edge,
           sr_solution.x_loadcap, sr_solution.P_A_load, sr_solution.P_B_load, sr_solution.P_C_load, sr_solution.Q_A_load, sr_solution.Q_B_load, sr_solution.Q_C_load,
           sr_solution.x_node,
           sr_solution.x_ess_ch, sr_solution.x_ess_disch,
           sr_solution.P_A_ess_ch, sr_solution.P_B_ess_ch, sr_solution.P_C_ess_ch,
           sr_solution.Q_A_ess_ch, sr_solution.Q_B_ess_ch, sr_solution.Q_C_ess_ch,
           sr_solution.P_A_ess_disch, sr_solution.P_B_ess_disch, sr_solution.P_C_ess_disch,
           sr_solution.Q_A_ess_disch, sr_solution.Q_B_ess_disch, sr_solution.Q_C_ess_disch,
           sr_solution.SOC_A, sr_solution.SOC_B, sr_solution.SOC_C,
           sr_solution.rh_setup, sr_solution.sr_setup]

    pickle.dump(obj, open("output/solution.dat", "wb"), True)

    return sr_solution


if __name__ == '__main__':
    # Set up the rolling horizon (rh) method parameters
    rh_start_time = "13:00"
    rh_horizon = 6  # total steps in each iteration
    rh_control = 6  # within each iteration, how many steps to carry out
    rh_set_step = 6  # steps set by the user
    rh_step_length = 1 # in minutes
    rh_iteration = int(np.ceil((rh_set_step-rh_horizon)/(rh_control-1) + 1))  # according to (control - 1) * (k - 1) + horizon >= total_step
    rh_total_step = (rh_control-1)*(rh_iteration-1) + rh_horizon  # total steps used by the algorithm,  according to (control - 1) * (k - 1) + horizon

    st_pf_balance        = True
    st_pf_voltage        = st_pf_balance
    st_voltage_limit     = True
    st_line_capacity     = True
    st_gen_capacity      = True
    st_gen_reserve       = True
    st_gen_ramp          = True
    st_gen_stepload      = True
    st_gen_unbalance     = True
    st_non_shed      = True
    st_topology = True
    st_connectivity  = st_topology
    st_sequence      = st_topology

    model_config = {"st_pf_balance": st_pf_balance,
                    "st_pf_voltage": st_pf_balance,
                    "st_voltage_limit": st_voltage_limit,
                    "st_line_capacity": st_line_capacity,
                    "st_gen_capacity": st_gen_capacity,
                    "st_gen_reserve": st_gen_reserve,
                    "st_gen_ramp": st_gen_ramp,
                    "st_gen_stepload": st_gen_stepload,
                    "st_gen_unbalance": st_gen_unbalance,
                    "st_non_shed": st_non_shed,
                    "st_connectivity": st_connectivity,
                    "st_sequence": st_sequence,
                    "st_topology": st_topology}
    rh_setup = {'rh_start_time': rh_start_time,
                'rh_horizon': rh_horizon,
                'rh_control': rh_control,
                'rh_set_step': rh_set_step,
                'rh_step_length': rh_step_length,
                'rh_iteration': rh_iteration,
                'rh_total_step': rh_total_step,
                'rh_model_config': model_config}

    # Set up the model environment
    sr_clpu_enable = True  # enable Cold Load Pick Up load model
    sr_re_enable = False  # enable considering renewable energies
    sr_es_enable = True  # enable considering ESS model
    sr_rg_enable = True  # enable considering voltage regulator
    sr_cap_enable = True  # enable considering capacitor banks

    sr_Vbase = 4160  # L-L voltage
    sr_Sbase = 1000  # 1 kVA
    sr_n_polygon = 4  # number of polygen to approximate x^2 + y^2 <= C
    sr_Vsrc = 1.0  # expected voltage in per unit of the black-start DG
    sr_M = 10000  # value used in the big-M method.
    sr_reserve_margin = 0.15  # capacity margin for each DG
    sr_setup = {'sr_clpu_enable': sr_clpu_enable,
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

    # Solve the SR problem, get the SR solution
    solve(rh_setup, sr_setup)


