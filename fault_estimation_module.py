#!/user/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import fault_location_import_module as fault_location_import_module
import pandas as pd

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


def find_sections(node_dict, fi_dict):
    # Thanks to the author of the algorthm: https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph/
    graph_node_dict = node_dict.copy()  # pay extra attention here
    for i in graph_node_dict:
        graph_node_dict[i] = graph_node(node_dict[i]["Node"])

    for i in fi_dict:
        if fi_dict[i]["FI"] == 0:
            graph_node_dict[fi_dict[i]['Node_A']].add_link(graph_node_dict[fi_dict[i]['Node_B']])

    graph_node_set = set()
    for i in graph_node_dict:
        graph_node_set.add(graph_node_dict[i])

    graph_islands = connected_components(graph_node_set)
    n_islands = 0
    islands_list = []
    for i in graph_islands:
        subislands = set()
        for j in i:
            subislands.update([j.name])  # not j.name, must be [j.name] as a string
        n_islands = n_islands + 1
        islands_list.append(subislands)


    return islands_list


def fault_location(islands_list, fi_dict):
    islands_dict = {}
    for island in islands_list:
        sublinks = []
        subfilinks = []
        subfistatus = []
        ingo = 0
        outgo = 0
        no_signal = 0
        fault_status = False
        for node in island:
            for k in fi_dict:
                a = fi_dict[k]['Node_A']
                b = node
                if node == fi_dict[k]['Node_A']:
                    if fi_dict[k]['Node_B'] in island: # k is in the same island, so A-B is an inside link
                        if k in sublinks:
                            pass
                        else:
                            sublinks.append(k)
                    else: # A-B is not inside the same island, it should be a line installed with FI, from A to B
                        if fi_dict[k]['FI'] == 1: # double check if FI is installed
                            if k in subfilinks:
                                pass
                            else:
                                subfilinks.append(k)
                                subfistatus.append(fi_dict[k]['Status'])
                                if fi_dict[k]['Status'] == 1:
                                    outgo += 1
                                elif fi_dict[k]['Status'] == -1:
                                    ingo += 1
                                else:
                                    no_signal += 1


                if node == fi_dict[k]['Node_B']:
                    if fi_dict[k]['Node_A'] in island: # k is in the same island, so A-B is an inside link
                        if k in sublinks:
                            pass
                        else:
                            sublinks.append(k)
                    else: # A-B is not inside the same island, it should be a line installed with FI, from A to B
                        if fi_dict[k]['FI'] == 1: # double check if FI is installed
                            if k in subfilinks:
                                pass
                            else:
                                subfilinks.append(k)
                                subfistatus.append(fi_dict[k]['Status'])
                                if fi_dict[k]['Status'] == 1:
                                    ingo += 1
                                elif fi_dict[k]['Status'] == -1:
                                    outgo += 1
                                else:
                                    no_signal += 1

        # determine if this island has fault inside based on the rule: ingo >= 1 AND outgo == 0
        if ingo >= 1 and outgo == 0:
            fault_status = True

            print('Faulted Section Found. Possible Faulted Lines:')
            faulted_lines = list(set().union(sublinks, subfilinks))
            for fl in faulted_lines:
                print(fl)
        else:
            fault_status = False

        islands_dict[node] = { 'Nodes':island,
                               'Lines': sublinks,
                               'FI_Lines': subfilinks,
                               'FI_status':subfistatus,
                               'FI_ingo': ingo,
                               'FI_outgo': outgo,
                               'FI_nosignal': no_signal,
                               'FI_fault': fault_status}


    return islands_dict



def weather_probability(node_dict, fi_dict, data_weather, delta_T):

    # determine which line is in which area
    for i in fi_dict:
        node_a = fi_dict[i]['Node_A']
        node_b = fi_dict[i]['Node_B']
        if node_dict[node_a]['Area'] == node_dict[node_b]['Area']:  # both ends of a line is in a same area
            a = data_weather[node_dict[node_a]['Area']] * fi_dict[i]['Length'] * 0.000189393939 * delta_T
            fi_dict[i]['Weather_Prob'] = 1 - np.exp(-1 * a)  # add a key word to fi_dict
        else:
            a = ( data_weather[node_dict[node_a]['Area']] + data_weather[node_dict[node_b]['Area']] ) / 2  * fi_dict[i]['Length'] * 0.000189393939 * delta_T
            fi_dict[i]['Weather_Prob'] = 1 - np.exp(-1 * a)

    return fi_dict


def fusion_fi_weather(fi_sections_dict, line_weather_failure_dict):
    line_failure_dict = line_weather_failure_dict
    # for each block, distribute the failure probability based on FI info to each line
    for i in fi_sections_dict:
        if fi_sections_dict[i]['FI_fault'] == True:
            n_line = len(fi_sections_dict[i]['Lines']) + len(fi_sections_dict[i]['FI_Lines'])
            fi_prob = fi_sections_dict[i]['FI_fault'] / n_line
            for j in fi_sections_dict[i]['Lines']:  # inside-block lines
                line_failure_dict[j]['FI_Prob'] = fi_prob
            for k in fi_sections_dict[i]['FI_Lines']:
                line_failure_dict[k]['FI_Prob'] = fi_prob
    for j in line_failure_dict:
        if 'FI_Prob' in line_failure_dict[j].keys():
            pass
        else:
            line_failure_dict[j]['FI_Prob'] = 0.0

    # fusion of FI failure prob and weather prob
    for k in line_failure_dict:
        fi_prob = line_failure_dict[k]['FI_Prob']
        we_prob = line_failure_dict[k]['Weather_Prob']
        line_failure_dict[k]['FI_Weather_Prob'] = 1 - (1 - fi_prob) * (1 - we_prob)


    return line_failure_dict


def main():
    # data read and process
    data_fi = fault_location_import_module.xlsx_read_fi()
    data_node = fault_location_import_module.construct_node()
    data_weather = fault_location_import_module.xlsx_read_weather()

    fi_dict = data_fi[3]
    node_dict = data_node[3]

    # prolong lines
    L = 100
    for i in fi_dict:
        fi_dict[i]['Length'] = fi_dict[i]['Length'] * L


    # develop fi_sections
    fi_sections_list = find_sections(node_dict, fi_dict)

    # fault location
    # Note the fault information is generated in MATLAB by running the MATLAB code calling OpenDSS engine
    fi_sections_dict = fault_location(fi_sections_list, fi_dict)

    # weather failure probability calculation
    delta_T = 40  # hours
    line_weather_failure_dict = weather_probability(node_dict, fi_dict, data_weather, delta_T)

    # fusion of FI data and weather failure prob data
    line_failure_dict = fusion_fi_weather(fi_sections_dict, line_weather_failure_dict)




    # Create a Pandas dataframe from the data.
    Data2write_nodeA = []
    Data2write_nodeB = []
    Data2write_fi_prob = []
    Data2write_weather_prob = []
    Data2write_fi_weather_prob = []

    for i in line_failure_dict:
        Data2write_nodeA.append(line_failure_dict[i]['Node_A'])
        Data2write_nodeB.append(line_failure_dict[i]['Node_B'])
        Data2write_fi_prob.append(line_failure_dict[i]['FI_Prob'])
        Data2write_weather_prob.append(line_failure_dict[i]['Weather_Prob'])
        Data2write_fi_weather_prob.append(line_failure_dict[i]['FI_Weather_Prob'])

    df = pd.DataFrame({'Node A': Data2write_nodeA, 'Node B': Data2write_nodeB, 'FI Prob': Data2write_fi_prob, 'Weather Prob': Data2write_weather_prob, 'FI+Weather Prob': Data2write_fi_weather_prob})
    index = ['Node A', 'Node B', 'FI Prob', 'Weather Prob', 'FI+Weather Prob']
    df = df[index]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('output/fault_location_data.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


    print('\n')
    
    
if __name__ == "__main__":
    main()

'''
if __name__ == "__main__":
    # data read and process
    data_fi = fault_location_import_module.xlsx_read_fi()
    data_node = fault_location_import_module.construct_node()
    data_weather = fault_location_import_module.xlsx_read_weather()

    fi_dict = data_fi[3]
    node_dict = data_node[3]

    # prolong lines
    L = 100
    for i in fi_dict:
        fi_dict[i]['Length'] = fi_dict[i]['Length'] * L


    # develop fi_sections
    fi_sections_list = find_sections(node_dict, fi_dict)

    # fault location
    # Note the fault information is generated in MATLAB by running the MATLAB code calling OpenDSS engine
    fi_sections_dict = fault_location(fi_sections_list, fi_dict)

    # weather failure probability calculation
    delta_T = 40  # hours
    line_weather_failure_dict = weather_probability(node_dict, fi_dict, data_weather, delta_T)

    # fusion of FI data and weather failure prob data
    line_failure_dict = fusion_fi_weather(fi_sections_dict, line_weather_failure_dict)




    # Create a Pandas dataframe from the data.
    Data2write_nodeA = []
    Data2write_nodeB = []
    Data2write_fi_prob = []
    Data2write_weather_prob = []
    Data2write_fi_weather_prob = []

    for i in line_failure_dict:
        Data2write_nodeA.append(line_failure_dict[i]['Node_A'])
        Data2write_nodeB.append(line_failure_dict[i]['Node_B'])
        Data2write_fi_prob.append(line_failure_dict[i]['FI_Prob'])
        Data2write_weather_prob.append(line_failure_dict[i]['Weather_Prob'])
        Data2write_fi_weather_prob.append(line_failure_dict[i]['FI_Weather_Prob'])

    df = pd.DataFrame({'Node A': Data2write_nodeA, 'Node B': Data2write_nodeB, 'FI Prob': Data2write_fi_prob, 'Weather Prob': Data2write_weather_prob, 'FI+Weather Prob': Data2write_fi_weather_prob})
    index = ['Node A', 'Node B', 'FI Prob', 'Weather Prob', 'FI+Weather Prob']
    df = df[index]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('output/fault_location_data.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


    print('\n')
'''