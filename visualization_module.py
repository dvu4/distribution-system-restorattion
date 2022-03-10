#!/user/bin/env python3
# -*- coding: utf-8 -*-

" This is the main program for running the optimization algorithm \
  Reference: [1] Bo Chen's MATLAB Code "

# [09/04/2017] Visualize the solution at each step
# [09/12/2017] Visualize the load demand, read solution files.
# [09/19/2017] Visualize through GUI interfaces
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
__date__ = "09/04/2017"  # Starting date

import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import restoration_module as sr


def solution_import():
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    myfile = os.path.join(data_folder, "solution.dat")
    sr_solution = pickle.load(open(myfile, "rb"))

    #sr_solution = pickle.load(open("output/solution.dat", "rb"))

    gen = sr_solution[0]
    P_A_gen = sr_solution[1]
    P_B_gen = sr_solution[2]
    P_C_gen = sr_solution[3]
    Q_A_gen = sr_solution[4]
    Q_B_gen = sr_solution[5]
    Q_C_gen = sr_solution[6]

    edge = sr_solution[7]

    load = sr_solution[8]
    P_A_load = sr_solution[9]
    P_B_load = sr_solution[10]
    P_C_load = sr_solution[11]
    Q_A_load = sr_solution[12]
    Q_B_load = sr_solution[13]
    Q_C_load = sr_solution[14]

    node = sr_solution[15]

    ess_ch = sr_solution[16]
    ess_disch = sr_solution[17]
    P_A_ess_ch = sr_solution[18]
    P_B_ess_ch = sr_solution[19]
    P_C_ess_ch = sr_solution[20]
    Q_A_ess_ch = sr_solution[21]
    Q_B_ess_ch = sr_solution[22]
    Q_C_ess_ch = sr_solution[23]
    P_A_ess_disch = sr_solution[24]
    P_B_ess_disch = sr_solution[25]
    P_C_ess_disch = sr_solution[26]
    Q_A_ess_disch = sr_solution[27]
    Q_B_ess_disch = sr_solution[28]
    Q_C_ess_disch = sr_solution[29]
    SOC_A = sr_solution[30]
    SOC_B = sr_solution[31]
    SOC_C = sr_solution[32]

    rh_setup = sr_solution[33]
    sr_setup = sr_solution[34]

    rh_start_time = rh_setup['rh_start_time']
    rh_horizon = rh_setup['rh_horizon']  # total steps in each iteration
    rh_control = rh_setup['rh_control']  # within each iteration, how many steps to carry out
    rh_set_step = rh_setup['rh_set_step']  # steps set by the user
    rh_step_length = rh_setup['rh_step_length']
    rh_iteration = rh_setup['rh_iteration']
    rh_total_step = rh_setup['rh_total_step']  # total steps used by the algorithm

    sr_clpu_enable = sr_setup['sr_clpu_enable']  # enable Cold Load Pick Up load model
    sr_re_enable = sr_setup['sr_re_enable']  # enable considering renewable energies
    sr_es_enable = sr_setup['sr_es_enable']  # enable considering ESS model
    sr_rg_enable = sr_setup['sr_rg_enable']  # enable considering voltage regulator
    sr_Vbase = sr_setup['sr_Vbase']
    sr_Sbase = sr_setup['sr_Sbase']
    sr_cap_enable = sr_setup['sr_cap_enable']  # enable considering capacitor bank
    sr_n_polygon = sr_setup['sr_n_polygon']  # number of polygen to approximate x^2 + y^2 <= C
    sr_Vsrc = sr_setup['sr_Vsrc']  # expected voltage of the black-start DG
    sr_M = sr_setup['sr_M']  # value used in the big-M method.
    sr_reserve_margin = sr_setup['sr_reserve_margin']

    return gen, P_A_gen, P_B_gen, P_C_gen, Q_A_gen, Q_B_gen, Q_C_gen, \
           edge, \
           load, P_A_load, P_B_load, P_C_load, Q_A_load, Q_B_load, Q_C_load, \
           node, \
           ess_ch, ess_disch, \
           P_A_ess_ch, P_B_ess_ch, P_C_ess_ch, Q_A_ess_ch, Q_B_ess_ch, Q_C_ess_ch, \
           P_A_ess_disch, P_B_ess_disch, P_C_ess_disch, Q_A_ess_disch, Q_B_ess_disch, Q_C_ess_disch, \
           SOC_A, SOC_B, SOC_C, \
           rh_start_time, rh_horizon, rh_control, rh_set_step, rh_step_length, rh_iteration, rh_total_step, \
           sr_clpu_enable, sr_re_enable, sr_es_enable, sr_rg_enable, sr_Vbase, sr_Sbase, sr_cap_enable, \
           sr_n_polygon, sr_Vsrc, sr_M, sr_reserve_margin

def plot_sequence(step_select):
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
     n_ess, ess_list, ess_set, ess_dict] = sr.data_import()

    ######################################################################################
    [gen, P_A_gen, P_B_gen, P_C_gen, Q_A_gen, Q_B_gen, Q_C_gen, \
    edge, \
    load, P_A_load, P_B_load, P_C_load, Q_A_load, Q_B_load, Q_C_load, \
    node, \
    ess_ch, ess_disch, \
    P_A_ess_ch, P_B_ess_ch, P_C_ess_ch, Q_A_ess_ch, Q_B_ess_ch, Q_C_ess_ch,
    P_A_ess_disch, P_B_ess_disch, P_C_ess_disch, Q_A_ess_disch, Q_B_ess_disch, Q_C_ess_disch, \
    SOC_A, SOC_B, SOC_C, \
    rh_start_time, rh_horizon, rh_control, rh_set_step, rh_step_length, rh_iteration, rh_total_step, \
    sr_clpu_enable, sr_re_enable, sr_es_enable, sr_rg_enable, sr_Vbase, sr_Sbase, sr_cap_enable, \
    sr_n_polygon, sr_Vsrc, sr_M, sr_reserve_margin] = solution_import()

    ######################################################################################
    # Plot single-line diagram of sequences
    t = step_select
    plt.figure()

    # set the y-limits of the current axes --> https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ylim.html
    plt.xlim(xmax=70)
    plt.ylim(ymax=45)

    G = nx.Graph()

    for i in range(n_node):
        if int(node[t, i]) == 1:
            G.add_node(node_list[i].Node)

    for e in range(n_edge):
        if int(edge[t, e]) == 1:
            if edge_list[e].Control == True:
                G.add_edge(edge_list[e].Node_A, edge_list[e].Node_B, color = 'r')
            else:
                G.add_edge(edge_list[e].Node_A, edge_list[e].Node_B, color='g')

    graph_edges = G.edges()
    graph_colors = [G[u][v]['color'] for u, v in graph_edges]

    pos = {}
    for n in node_list:
        pos.update({n.Node: n.GIS})

    nx.draw_networkx_labels(G, pos, font_size= 6)
    nx.draw_networkx_edges(G, pos, edges=graph_edges, edge_color=graph_colors)
    nx.draw_networkx_nodes(G, pos, node_size = 60 )

    #plt.axis('off')
    #plt.savefig("diagram_step.png")
    #plt.savefig("output/diagram_step.png")

    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    myfile = os.path.join(data_folder, "diagram_step.png")
    plt.savefig(myfile)
    #https://networkx.github.io/documentation/latest/_modules/networkx/drawing/layout.html
    #http://sparkandshine.net/en/networkx-application-notes-draw-a-graph-with-matplotlib/


def plot_load(wanted_node):
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
     n_ess, ess_list, ess_set, ess_dict] = sr.data_import()

    ######################################################################################
    [gen, P_A_gen, P_B_gen, P_C_gen, Q_A_gen, Q_B_gen, Q_C_gen, \
    edge, \
    load, P_A_load, P_B_load, P_C_load, Q_A_load, Q_B_load, Q_C_load, \
    node, \
    ess_ch, ess_disch, \
    P_A_ess_ch, P_B_ess_ch, P_C_ess_ch, Q_A_ess_ch, Q_B_ess_ch, Q_C_ess_ch,
    P_A_ess_disch, P_B_ess_disch, P_C_ess_disch, Q_A_ess_disch, Q_B_ess_disch, Q_C_ess_disch, \
    SOC_A, SOC_B, SOC_C, \
    rh_start_time, rh_horizon, rh_control, rh_set_step, rh_step_length, rh_iteration, rh_total_step, \
    sr_clpu_enable, sr_re_enable, sr_es_enable, sr_rg_enable, sr_Vbase, sr_Sbase, sr_cap_enable, \
    sr_n_polygon, sr_Vsrc, sr_M, sr_reserve_margin] = solution_import()

    ######################################################################################
    # Plot load demand during restoration
    for n in range(n_loadcap):
        if loadcap_list[n].Node == wanted_node:
            load_number = n

            plt.figure()
            plt.subplot(3,1,1)
            index = np.arange(len(P_A_load))
            values = P_A_load[:,load_number]
            bar_width = 0.5
            opacity = 0.4
            bar_color = 'b'
            error_config = {'ecolor': '0.3'}
            bar_label = loadcap_list[load_number].Name

            plt.bar(index, values, bar_width,
                    alpha=opacity,
                    color=bar_color,
                    error_kw=error_config,
                    label=bar_label)

            plt.subplot(3, 1, 2)
            index = np.arange(len(P_B_load))
            values = P_B_load[:, load_number]
            bar_width = 0.5
            opacity = 0.4
            bar_color = 'y'
            error_config = {'ecolor': '0.3'}
            bar_label = loadcap_list[load_number].Name

            plt.bar(index, values, bar_width,
                    alpha=opacity,
                    color=bar_color,
                    error_kw=error_config,
                    label=bar_label)

            plt.subplot(3, 1, 3)
            index = np.arange(len(P_C_load))
            values = P_C_load[:, load_number]
            bar_width = 0.5
            opacity = 0.4
            bar_color = 'g'
            error_config = {'ecolor': '0.3'}
            bar_label = loadcap_list[load_number].Name

            plt.bar(index, values, bar_width,
                    alpha=opacity,
                    color=bar_color,
                    error_kw=error_config,
                    label=bar_label)

            current_path = os.getcwd()
            data_folder = os.path.join(current_path,"output")
            myfile = os.path.join(data_folder, "load_profile.png")
            plt.savefig(myfile)
            #plt.savefig("output/load_profile.png")

            



if __name__ == '__main__':
    # plot_sequence(4)
    plot_sequence(0)
    # plot_load('49')
    plot_load('16')
    print('Done')