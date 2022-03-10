import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set()



def read_pickle(data_pickle):
    with open(data_pickle, 'rb') as f:
        datadict = pickle.load(f)
    return datadict


def import_dss_data(data_pickle):

    obj = read_pickle(data_pickle)
    AllBusNames = obj[0]
    AllLoadNames = obj[1]
    AllLineNames = obj[2]
    AllTransNames = obj[3]
    AllCapacitorNames = obj[4]
    AllTransNames = obj[5]
    AllSubNames = obj[6]
    Circuit = obj[7]

    return AllBusNames , AllLoadNames, AllLineNames, AllTransNames,  AllCapacitorNames, AllTransNames, AllSubNames, Circuit


def generate_network_data(data_pickle):

    AllBusNames , AllLoadNames, AllLineNames, AllTransNames,  AllCapacitorNames, AllTransNames, AllSubNames, Circuit = import_dss_data(data_pickle)
    



    listBusKeys = list(AllBusNames.keys())
    listBuses = []
    for i in listBusKeys[3:]:
        if AllBusNames[i]['Coorddefined'] == True:
            listBuses.append(AllBusNames[i])


    listSubKeys = list(AllSubNames.keys())
    listSubs = []
    for i in listSubKeys[3:]:
        if AllSubNames[i]['Coorddefined'] == True:
            listSubs.append(AllSubNames[i])


    listLineKeys = list(AllLineNames.keys())
    listLines = []
    for i in listLineKeys[3:]:
        listLines.append((AllLineNames[i]['Bus1'], AllLineNames[i]['Bus2']))


    listHighLines = []
    listLowLines = []

    listLowBuses = []
    listHighBuses = []

    voltageList = []
    for i in listBuses:

        kV_List = ['kVBase', 'kV_LN']

        if kV_List[0] in i.keys():
            voltageList.append(i[kV_List[0]])
        elif kV_List[1] in i.keys():
            voltageList.append(i[kV_List[1]])
    
    voltageList = list(set(voltageList))

    if len(voltageList) == 2:

        for i in listBuses:
            kV_List = ['kVBase', 'kV_LN']

            if kV_List[0] in i.keys():
                if i[kV_List[0]]*np.sqrt(3) == 34.5:
                    listHighBuses.append(i)
                elif i[kV_List[0]]*np.sqrt(3) == 13.2:
                    listLowBuses.append(i)

            elif kV_List[1] in i.keys():
                if i[kV_List[1]]*np.sqrt(3) == 4.16:
                    listHighBuses.append(i)
                elif i[kV_List[1]]*np.sqrt(3) == 0.48:
                    listLowBuses.append(i)

   
        lowBusNames = []
        highBusNames = []

        lowBusNames = [i['Name'] for i in listLowBuses]
        lowBusNames = list(set(lowBusNames))

        highBusNames = [i['Name'] for i in listHighBuses]
        highBusNames = list(set(highBusNames))

        listHighLines = [i for i in listLines if ((i[0] in highBusNames) &  (i[1] in highBusNames))]
        listLowLines = [i for i in listLines if ((i[0] in lowBusNames) &  (i[1] in lowBusNames))]

    return listBuses, listHighBuses, listLowBuses, listSubs, listLines, listHighLines, listLowLines, voltageList

    
    
def make_proxy(clr, mappable, **kwargs):
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)


def plot_topological_distribution_networks(data_pickle):

    data_name = data_pickle.split(os.sep)
    data_name = data_name[-1]

    #listBuses, listSubs, listLines = generate_network_data(data_pickle)
    listBuses, listHighBuses, listLowBuses, listSubs, listLines, listHighLines, listLowLines, voltageList = generate_network_data(data_pickle)
    
    plt.figure(figsize=(15,15))
    # set the y-limits of the current axes --> https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ylim.html
    #plt.xlim(xmax=70)
    #plt.ylim(ymax=45)


    if (not listHighBuses) and (not listLowBuses) and (not listHighLines) and (not listLowLines): 
        bus_nodes = [listBuses[i]['Name'] for i in range(len(listBuses))]
        sub_nodes = [listSubs[i]['Name'] for i in range(len(listSubs))]

        pos = {}
        for n in listBuses:
            pos.update({n['Name']: (n['Coord_X'], n['Coord_Y'])})

        for n in listSubs:
            pos.update({n['Name']: (n['Coord_X'], n['Coord_Y'])})


        G = nx.Graph()
        G.add_nodes_from(bus_nodes, Type='BUS')
        G.add_nodes_from(sub_nodes, Type='SUBSTATION')

        # extract nodes with specific setting of the attribute
        bus_nodes = [n for (n,ty) in nx.get_node_attributes(G,'Type').items() if ty == 'BUS']
        sub_nodes = [n for (n,ty) in nx.get_node_attributes(G,'Type').items() if ty == 'SUBSTATION']

        n_edge = len(listLines)
        edge_list = listLines
        

        for e in range(n_edge):
            if (edge_list[e][0] in bus_nodes) and (edge_list[e][1] in bus_nodes):
                G.add_edge(edge_list[e][0], edge_list[e][1], color = 'dodgerblue', weight=6)
    
        graph_edges = G.edges()
        graph_colors = [G[u][v]['color'] for u, v in graph_edges]
        graph_weights = [G[u][v]['weight'] for u,v in graph_edges]
        #print(G.number_of_edges())

        # now draw them in subsets  using the `nodelist` arg
        nx.draw_networkx_nodes(G, pos, nodelist = bus_nodes, node_size = 10, node_color='honeydew', node_shape='o') # ‘so^>v<dph8’       
        nx.draw_networkx_nodes(G, pos, nodelist = sub_nodes, node_size = 400, node_color='red', node_shape='^')
        h = nx.draw_networkx_edges(G, pos, edges=graph_edges, edge_color=graph_colors, width=graph_weights, edge_cmap=plt.cm.Set2)
        

        # generate proxies with the above function
        proxies = [make_proxy(clr, h, lw=2) for clr in list(set(graph_colors))]
        # and some text for the legend -- you should use something from df.
        labels = ["kV Base {} kV".format(k) for k in voltageList]
        plt.legend(proxies, labels, prop={'size': 20})


    else:
        high_bus_nodes = [listHighBuses[i]['Name'] for i in range(len(listHighBuses))]
        low_bus_nodes  = [listLowBuses[i]['Name'] for i in range(len(listLowBuses))]

        sub_nodes = [listSubs[i]['Name'] for i in range(len(listSubs))]

        pos = {}
        for n in listHighBuses:
            pos.update({n['Name']: (n['Coord_X'], n['Coord_Y'])})
    
        for n in listLowBuses:
            pos.update({n['Name']: (n['Coord_X'], n['Coord_Y'])})
    
        for n in listSubs:
            pos.update({n['Name']: (n['Coord_X'], n['Coord_Y'])})


        G = nx.Graph()
        G.add_nodes_from(high_bus_nodes, Type='HIGH_BUS')
        G.add_nodes_from(low_bus_nodes, Type='LOW_BUS')
        G.add_nodes_from(sub_nodes, Type='SUBSTATION')


        n_high_edge = len(listHighLines)
        high_edge_list = listHighLines

        for e in range(n_high_edge):
            if (high_edge_list[e][0] in high_bus_nodes) and (high_edge_list[e][1] in high_bus_nodes):
                G.add_edge(high_edge_list[e][0], high_edge_list[e][1], color = 'dodgerblue', weight=6)
        

        n_low_edge = len(listLowLines)
        low_edge_list = listLowLines

        for e in range(n_low_edge):
            if (low_edge_list[e][0] in low_bus_nodes) and (low_edge_list[e][1] in low_bus_nodes):
                G.add_edge(low_edge_list[e][0], low_edge_list[e][1], color = 'limegreen', weight=6)


        graph_edges = G.edges()
        graph_colors = [G[u][v]['color'] for u, v in graph_edges]
        graph_weights = [G[u][v]['weight'] for u,v in graph_edges]


        # now draw them in subsets  using the `nodelist` arg
        nx.draw_networkx_nodes(G, pos, nodelist = high_bus_nodes, node_size = 10, node_color='honeydew', node_shape='o') # ‘so^>v<dph8’
        nx.draw_networkx_nodes(G, pos, nodelist = low_bus_nodes, node_size = 10, node_color='orangered', node_shape='o')
        nx.draw_networkx_nodes(G, pos, nodelist = sub_nodes, node_size = 400, node_color='red', node_shape='^')
        h = nx.draw_networkx_edges(G, pos, edges=graph_edges, edge_color=graph_colors, width=graph_weights, edge_cmap=plt.cm.Set2)

        voltageList = [i*np.sqrt(3) for i in voltageList]

        # generate proxies with the above function
        proxies = [make_proxy(clr, h, lw=2) for clr in list(set(graph_colors))]
        # and some text for the legend -- you should use something from df.
        labels = ["kV Base {} kV".format(k) for k in voltageList]
        plt.legend(proxies, labels, prop={'size': 20})


    topologyFile = str(data_name[:-4])

    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    file_to_save = str(data_name) + "_topology.png"
    topologyFilePath = os.path.join(data_folder, file_to_save)

    plt.savefig(topologyFilePath)
    return topologyFile, topologyFilePath 