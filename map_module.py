

__author__ = "Duc Vu"
__copyright__ = "Copyright 2017, " \
                "The GMLC Project: A Closed-Loop Distribution System Restoration Tool" \
                " for Natural Disaster Recovery"
__maintainer__ = "Duc Vu"
__email__ = "ducvuchicago@gmail.com"


# # Power grid with Folium
# Import the necessary Python modules
import os
import sys
sys.path.insert(0,'..')
import folium
import folium.plugins as plugins
from folium.plugins import MarkerCluster
from folium import IFrame, Map, Marker, GeoJson, LayerControl

#print(folium.__file__)
#print(folium.__version__)

import pandas as pd
import numpy as np
import random
from random import randint

import re

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from colormap import rgb2hex

import base64
from io import BytesIO

#! conda install -c anaconda networkx
import networkx as nx

#! conda install -c anaconda seaborn 
import seaborn as sns
sns.set()

from IPython.display import IFrame

import json
from json2html import *

import pyproj


from fault_estimation_module import *

def calculate_failure_estimation():
    main()
    
calculate_failure_estimation()

'''
from glob import glob
for filename in glob("*.html"):
    if os.path.exists(filename):
        os.remove(filename)

if os.path.exists('networkx.png'): 
    os.remove('networkx.png')
'''


# #  Helper functions
# ### a. Convert X/Y to lat/long
def conv_xy_to_latlon(xy_list, origin_lat_lon):
    # lat_1=40.666667, lat_2=41.033333,
    origin_lat, origin_lon = origin_lat_lon
    lats, lons = [], []
    
    for x, y in xy_list:
        pnyc = pyproj.Proj( proj='tmerc', datum='NAD83', 
                           lat1=41.7557, lat2=41.6860,
                           lat_0=origin_lat, lon_0=origin_lon, 
                           x_0=0, y_0=0.0)
    
        lon, lat = pnyc(x, y, inverse=True)
        #print(lat, lon)
        lats.append(lat)
        lons.append(lon) 
        
    return lats, lons


# ### b. Convert probability to color
def prob_to_color(prob, cmapcolor = 'Reds'):
    
    cmap = cm.get_cmap(cmapcolor)
    
    #rgba = [(0, 255, 0, 255) if i < threshold_prob else cmap(i,bytes=True) for i in prob]
    rgba = [(0, 255, 0, 255) if i < 0.01 else cmap(i,bytes=True) for i in prob]
    #rgba = [cmap(i,bytes=True) for i in prob]
    hex_color = [rgb2hex(i[0], i[1], i[2]) for i in rgba]
    
    return hex_color



# ### c. Find the geometry points
def find_geometry_points(df_node, df_egde):
    #node_label_dict = dict([(node,(lat, lon)) for node, lat, lon in zip(df_node['Node'], df_node['lat'],df_node['lon'])])
    node_label_dict = {x[0]: x[3:] for x in df_node.itertuples(index=False)}

    node_a = [node_label_dict[i] for i in df_egde['Node A'].tolist()]
    node_b = [node_label_dict[i] for i in df_egde['Node B'].tolist()]
    node_ab = list(zip(node_a,node_b))
    df_egde['Node_A_coord'] = node_a
    df_egde['Node_B_coord'] = node_b
    df_egde['Node_AB_coord'] = node_ab
    return df_egde


# ### d. Calculate the center point of latitude/longitude coordinate pairs
def find_centroid(df_egde):
    
    centroid_coords = lambda x: [sum(y) / (len(y)) for y in zip(*x)]
    df_egde['centroid_Node_AB_coord'] = df_egde['Node_AB_coord'].apply(centroid_coords)
    df_egde['lat'] = df_egde['centroid_Node_AB_coord'].apply(lambda x: x[0])
    df_egde['lon'] = df_egde['centroid_Node_AB_coord'].apply(lambda x: x[1]) 
    
    return df_egde


# ### e. Convert dataframe to html
def df_to_html(row, info_name = None):
    
    style = '<style> #myDIV { width: 360px;height: 50px;background-color: #FF7006;color: white;text-align:center;font-family:courier;font-size:10px;} </style> <div id="myDIV"> <h1>' + str(info_name) + '</h1> </div>' 
    newdf = pd.DataFrame(row).copy(deep=True)

    html = newdf.to_html()
    html = html.replace('\n', ' ')
    html = html.replace('<table border="1" class="dataframe">','<table class="table table-striped">')
    html =  style + html


    #remove content between <thead>  </thead>
    pattern = r'.*?\<thead>(.*)\</thead>.*'
    match = re.search(pattern, html)
    header = match.group(1)
    html = html.replace(header,'')


    #replace <tr> with <tr style="text-align: left;">
    html = html.replace('<tr>','<tr style="text-align: left;">')
    return html


# ### f. Show popup on mouse-over
import re
import fileinput

def find_marker(html, marker_type=True):
    with open(html) as inf:
        txt = inf.read()

    if marker_type==True:
        #Find all the markers names given by folium
        markers = re.findall(r'\bmarker_\w+', txt)
        markers = list(set(markers))
    else:
        #Find all the polyline names given by folium
        polylines = re.findall(r'\bpoly_line_\w+', txt) 
        markers = list(set(polylines))

    return markers


def popup_mouseover(html, marker):
    for linenum,line in enumerate( fileinput.FileInput(html,inplace=1) ):
        pattern  = marker + ".bindPopup"
        pattern2 = marker + ".on('mouseover', function (e) {this.openPopup();});"
        pattern3 = marker + ".on('mouseout', function (e) {this.closePopup();});"
    
        if pattern in line:
            print(line.rstrip())
            print(pattern2)
            print(pattern3)
        else:
            print(line.rstrip())
    return html


## 1. Preprocess Data
#### a. Read Nodes in Test Feeder Data

#def read_node_data():
#current_path = os.getcwd()
#filename = 'node_data_py.xls'
#xl = pd.ExcelFile(current_path + '/data/' + filename)

current_path = os.getcwd()
data_folder = os.path.join(current_path,"data")
filepath = os.path.join(data_folder, "node_data_py.xls")


xl = pd.ExcelFile(filepath)
xl.sheet_names

df_node = xl.parse("Sheet1")
df_node.head()

df_node.columns = ['Node', 'X', 'Y' ]
df_node  = df_node .iloc[2:]

df_node['Node'] = df_node['Node'].astype(str)
df_node  = pd.concat([df_node , pd.DataFrame(columns = ['lat','lon'])], sort=True)


df_node['Node'] = df_node['Node'].str.upper()
df_node[['X','Y']] = df_node[['X','Y']] *80


lat1, lon1 = 41.7557, -72.7831 
lat2, lon2 = 41.6860, -72.6890

hudson_coord = [ (lat1 + lat2)/2, (lon1 + lon2)/2 ]
loc_list = df_node[["X","Y"]].values.tolist()
lats, lons = conv_xy_to_latlon(loc_list, hudson_coord)

df_node['lat'] = lats
df_node['lon'] = lons


'''
node = df_node.iloc[:,0]
coord = df_node.iloc[:,3:]

labels = node.values
coordinates = coord.values
'''


#### b. Read Lines in Test Feeder Data 
#def read_line_data(df_node):
#current_path = os.getcwd()
#filename = 'line_data_py.xls'
#xl = pd.ExcelFile(current_path + '/data/' + filename)

current_path = os.getcwd()
data_folder = os.path.join(current_path,"data")
filepath = os.path.join(data_folder, "line_data_py.xls")
xl = pd.ExcelFile(filepath)

xl.sheet_names

df_line = xl.parse("Sheet1")

df_line.columns = ['Node A', 'Node B', 'Length (ft.)', 'Config.', 'control', 'status', 'outage']
df_line = df_line.iloc[2:]

df_line['Node A'] = df_line['Node A'].astype(str)
df_line['Node B'] = df_line['Node B'].astype(str)

df_line['Node A'] = df_line['Node A'].str.upper()
df_line['Node B'] = df_line['Node B'].str.upper()
df_line = df_line[df_line['status'] != True]

df_line = find_geometry_points(df_node, df_line)


df_damaged_line = df_line[df_line['outage'] == True]
df_damaged_line = df_damaged_line[['Node A', 'Node B', 'outage', 'Node_AB_coord', 'Node_A_coord', 'Node_B_coord']]

df_damaged_line = find_centroid(df_damaged_line)

df_line = find_geometry_points(df_node, df_line)


df_damaged_line = df_line[df_line['outage'] == True]
#df_damaged_line = df_damaged_line[['outage', 'Node_AB_coord', 'Node_A_coord', 'Node_B_coord']]
    
centroid_coords = lambda x: [sum(y) / (len(y)) for y in zip(*x)]

df_damaged_line['centroid_Node_AB_coord'] = df_damaged_line['Node_AB_coord'].apply(centroid_coords)
df_damaged_line['centroid_Node_AB_coord'] = df_damaged_line['centroid_Node_AB_coord'].apply(lambda x: tuple(x))

df_damaged_line['new_coord'] = list(zip(df_damaged_line['Node_B_coord'], df_damaged_line['centroid_Node_AB_coord']))

df_damaged_line['quadroid_Node_AB_coord'] = df_damaged_line['new_coord'].apply(centroid_coords)
df_damaged_line['quadroid_Node_AB_coord'] = df_damaged_line['quadroid_Node_AB_coord'].apply(lambda x: tuple(x))


df_damaged_line['lat'] = df_damaged_line['centroid_Node_AB_coord'].apply(lambda x: x[0])
df_damaged_line['lon'] = df_damaged_line['centroid_Node_AB_coord'].apply(lambda x: x[1])
df_damaged_line1 = df_damaged_line[['Node A', 'Node B','Length (ft.)', 'Config.', 'control', 'status', 'outage', 'lat', 'lon']]


df_damaged_line['lat'] = df_damaged_line['quadroid_Node_AB_coord'].apply(lambda x: x[0])
df_damaged_line['lon'] = df_damaged_line['quadroid_Node_AB_coord'].apply(lambda x: x[1])
df_damaged_line2 = df_damaged_line[['Node A', 'Node B','Length (ft.)', 'Config.', 'control', 'status', 'outage', 'lat', 'lon']]


df_damaged_line1['lat'] - df_damaged_line2['lat']
    



# ### c. Read Failure Estimation data
def extract_damaged_node(df, damage_prob):
    df_damaged_line = df[df[damage_prob] >= 0.7]
    df_damaged_line = df[[damage_prob, 'Node_AB_coord']]
    
    centroid_coords = lambda x: [sum(y) / len(y) for y in zip(*x)]
    
    df_damaged_line['centroid_Node_AB_coord'] = df_damaged_line['Node_AB_coord'].apply(centroid_coords)
    df_damaged_line['lat'] = df_damaged_line['centroid_Node_AB_coord'].apply(lambda x: x[0])
    df_damaged_line['lon'] = df_damaged_line['centroid_Node_AB_coord'].apply(lambda x: x[1])
    df_damaged_line = df_damaged_line[['lat', 'lon']]
    return df_damaged_line


#def read_fault_data(df_node):
#current_path = os.getcwd()
#filename = 'fault_location_data.xlsx'
#xl = pd.ExcelFile(current_path + '/output/' + filename)

current_path = os.getcwd()
data_folder = os.path.join(current_path,"output")
filepath = os.path.join(data_folder, "fault_location_data.xlsx")
xl = pd.ExcelFile(filepath)

xl.sheet_names

df_failure_estimation = xl.parse("Sheet1")


## convert lower string to capital string 
df_failure_estimation['Node A'] = df_failure_estimation['Node A'].str.upper()
df_failure_estimation['Node B'] = df_failure_estimation['Node B'].str.upper()


##### convert probability to color
#threshold_prob = 0.001

FI_Prob_color = prob_to_color(df_failure_estimation['FI Prob'].tolist(), cmapcolor = 'OrRd')
Weather_Prob_color = prob_to_color(df_failure_estimation['Weather Prob'].tolist(), cmapcolor = 'OrRd')
FI_Weather_Prob_color = prob_to_color(df_failure_estimation['FI+Weather Prob'].tolist(), cmapcolor = 'OrRd')

df_failure_estimation['FI_Prob_color'] = FI_Prob_color
df_failure_estimation['Weather_Prob_color'] = Weather_Prob_color
df_failure_estimation['FI+Weather_Prob_color'] = FI_Weather_Prob_color

##### add pair of lat/lon coordinates 
df_failure_estimation = find_geometry_points(df_node, df_failure_estimation)


##### extract damaged line with different damage probabilities
df_damaged_line = df_failure_estimation[df_failure_estimation['Weather Prob'] >= 0.7]
df_damaged_line = df_damaged_line[['Weather Prob', 'Node_AB_coord']]

    
df_FI_damaged_line = extract_damaged_node(df_failure_estimation, 'FI Prob')
df_Weather_damaged_line = extract_damaged_node(df_failure_estimation, 'Weather Prob')
df_FI_Weather_damaged_line = extract_damaged_node(df_failure_estimation, 'FI+Weather Prob')
    
    

# ### d. Read fault indicator data
#def read_fault_indicator_data(df_node, df_line, df_failure_estimation):
#current_path = os.getcwd()
#filename = 'fault_indicator_data.xls'
#xl = pd.ExcelFile(current_path + '/data/' + filename)

current_path = os.getcwd()
data_folder = os.path.join(current_path,"data")
filepath = os.path.join(data_folder, "fault_indicator_data.xls")
xl = pd.ExcelFile(filepath)

xl.sheet_names

df_fault_indicator = xl.parse("Sheet1")


df_fault_indicator.columns = ['Node A', 'Node B', 'Length (ft.)', 'FI_Installation', 'FI_Direction']
df_fault_indicator  = df_fault_indicator.iloc[2:]

df_fault_indicator['Fault Indicator Status'] = np.where(df_fault_indicator['FI_Installation'] == 1, df_fault_indicator['FI_Direction'], 'N/A')


df_fault_indicator['Node A'] = df_fault_indicator['Node A'].astype(str)
df_fault_indicator['Node B'] = df_fault_indicator['Node B'].astype(str)

df_fault_indicator['Node A'] = df_fault_indicator['Node A'].str.upper()
df_fault_indicator['Node B'] = df_fault_indicator['Node B'].str.upper()


df_fault_indicator = find_geometry_points(df_node, df_fault_indicator)


df_fault_indicator_OFF = df_fault_indicator[df_fault_indicator['FI_Installation'] == 1]
df_fault_indicator_ON = df_fault_indicator[(df_fault_indicator['FI_Installation'] == 1) & (df_fault_indicator['FI_Direction'] == 1)]



#### Extract  locations  of OFF Fault Indicator 
df_fault_indicator_OFF = find_centroid(df_fault_indicator_OFF)
df_FI_OFF_location = df_fault_indicator_OFF[['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status']]



#### Extract  locations  of ON Fault Indicator 
df_fault_indicator_ON = find_centroid(df_fault_indicator_ON)
df_FI_ON_location = df_fault_indicator_ON[['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status']]



#### Merge df_line,  df_fault_indicator and df_failure_estimation.
df_merged_line = df_line.merge(df_fault_indicator, how = 'inner', on = ['Node A', 'Node B', 'Node_A_coord', 'Node_B_coord', 'Node_AB_coord','Length (ft.)'])


df_merged_line2 = df_merged_line.merge(df_failure_estimation, how = 'inner', on = ['Node A', 'Node B', 'Node_A_coord', 'Node_B_coord', 'Node_AB_coord'])
    





'''
### 2. Create network 
def generate_graph(df_node, df_edge):

    # Create empty graph
    g = nx.Graph()

    # Add edges and edge attributes
    for idx, row in df_edge.iterrows():
        g.add_edge(row[0], row[1], attr_dict=row[2:].to_dict())
        
    # Add node attributes
    for idx, row in df_node.iterrows():
        g.node[row['Node']].update(row[1:].to_dict())
    
    #Print out some summary statistics before visualizing the graph.
    print('# of edges: {}'.format(g.number_of_edges()))
    print('# of nodes: {}'.format(g.number_of_nodes()))
    
    return g

g = generate_graph(df_node, df_line)


#### a. Plot network 
# Define node positions data structure (dict) for plotting
node_positions = {node[0]: (node[1]['X'], node[1]['Y']) for node in g.nodes(data=True)}

# Preview of node_positions with a bit of hack (there is no head/slice method for dictionaries).
#print(dict(list(node_positions.items())[0:5]))


plt.figure(figsize=(10, 8))
nx.draw(g, pos=node_positions, node_size=500, node_color='red', with_labels=True)
plt.title('Graph Representation of Test feeder', size=15)
#networkx_path = '/Users/ducvu/Desktop/'
networkx_path = os.getcwd()
#networkx_path = '/Users/ducvu/Documents/Argonne_Restoration_Tool/restoration_tool_final/'
plt.savefig(networkx_path + "/networkx.png", format="PNG")
print('finish save networkx.png')
#plt.show()
'''


# # 3. Use Folium to Plot Data
'''
graph_centroid = 41.7557, -72.7831

#Create basemap specifying map center, zoom level, and using the default OpenStreetMap tiles
graph_map = folium.Map(location=graph_centroid, zoom_start = 11, control_scale = True, tiles='Stamen Terrain')

graph_map.fit_bounds([[41.7507, -72.7340], [41.7183, -72.6720]])


folium.TileLayer('stamentoner').add_to(graph_map)
folium.TileLayer('stamenwatercolor').add_to(graph_map)
folium.TileLayer('openstreetmap').add_to(graph_map)
folium.TileLayer('cartodbdark_matter').add_to(graph_map)
folium.TileLayer('cartodbpositron').add_to(graph_map)
'''

#### a. Create folium polyline layer
from folium import IFrame

# make folium polyline for each edge in df_edge
def generate_folium_polyline(edge, attr, edge_width, edge_opacity, prob_name = None, info_name = None):
    """
    Turn a row from the gdf_edges GeoDataFrame into a folium PolyLine 
    Parameters
    ----------
    edge : GeoSeries
        a row from the gdf_edges GeoDataFrame
    edge_width : numeric
        width of the edge lines
    edge_opacity : numeric
        opacity of the edge lines
    Returns
    -------
    pl : folium.PolyLine
    """
    locations = [(coord[0], coord[1]) for coord in edge['Node_AB_coord']]
    
    #edge color
    if prob_name != None:
        edge_color = edge[prob_name]
    else:
        edge_color = '#00FF7F' # green color
        
         
    row = edge[attr]
    html = df_to_html(row, info_name = info_name)
    iframe = IFrame(html=html, width=360, height=240)
    popup = folium.Popup(iframe, max_width=2650)
    
    
    # create a folium polyline with attributes
    pl = folium.PolyLine(locations=locations, popup=popup, color=edge_color, weight=edge_width, opacity=edge_opacity)
    return pl


#create folium polyline layer
def add_polyline(mapobj, df, attr = [], layer_name = 'Default', prob_name = None, info_name = None):
    """
    Add the layer of folium polylines from edges GeoDataFrame
    Parameters
    ----------
    mapobj : Folium map object
        Folium map object
    gdf_edges : GeoDataFrame
        edges GeoDataFrame
    Returns
    -------
    layer of folium polylines
    """
    
    #polyline_layer = folium.FeatureGroup(name = str(layer_name), overlay=False)
    polyline_layer = folium.FeatureGroup(name = str(layer_name))
    
    for _, row in df.iterrows():
        pl = generate_folium_polyline(row, attr, edge_width=5, edge_opacity=3, prob_name=prob_name, info_name = info_name)
        pl.add_to(polyline_layer)

    polyline_layer.add_to(mapobj)
    
    return mapobj



#### b. Create point cluster layer
def display_image_in_html(imgage):
    encoded_image = base64.b64encode(open(imgage, 'rb').read())
    decoded_image = base64.b64decode(encoded_image)
    image_url = BytesIO(decoded_image)
    return image_url


#create point cluster layer
def add_point_clusters(mapobj, df_node, attr = [], node_image = None, node_name = None, icon_size=(15, 15), info_name = None):
    """
    Add the layer of point locations from edges GeoDataFrame
    Parameters
    ----------
    mapobj : map object
        Folium map object
    gdf_nodes : GeoDataFrame
        nodes GeoDataFrame
    Returns
    -------
    layer of point locations
    """      

    #point_layer = folium.FeatureGroup(name = node_name, overlay=False)
    point_layer = folium.FeatureGroup(name = node_name)

    for _, row in df_node.iterrows(): 
        
        icon_url = node_image

        row = row[attr]
        html = df_to_html(row, info_name = info_name)
        iframe = IFrame(html=html, width=360, height=240)
        popup = folium.Popup(iframe, max_width=2650)
        
        icon = folium.features.CustomIcon(icon_url, icon_size=icon_size)
        
        # extract lat and long coordinates to assign to the marker
        folium.Marker(np.array([row.lat, row.lon]), popup=popup, icon=icon).add_to(point_layer)
        
    #Create a Folium feature group for this layer, since we will be displaying multiple layers
    point_layer.add_to(mapobj)
    return mapobj




### Generate 4 individual maps
### 1st map
def gen_map_1():
    locations = 41.7557, -72.7831
    bound_box = [[41.7507, -72.7340], [41.7183, -72.6720]]

    #Create basemap specifying map center, zoom level, and using the default OpenStreetMap tiles
    layer_map_1 = folium.Map(location = locations, zoom_start = 11, control_scale = True, tiles='Stamen Terrain')
    layer_map_1.fit_bounds(bound_box)

    #Update basemap with tile Layers
    folium.TileLayer('stamentoner').add_to(layer_map_1)
    folium.TileLayer('stamenwatercolor').add_to(layer_map_1)
    folium.TileLayer('openstreetmap').add_to(layer_map_1)
    folium.TileLayer('cartodbdark_matter').add_to(layer_map_1)
    folium.TileLayer('cartodbpositron').add_to(layer_map_1)


    #Update basemap with Polyline Layer
    layer_map_1 = add_polyline(layer_map_1, 
                               df_merged_line2, 
                               attr = ['Node A', 'Node B', 'Length (ft.)', 'Config.', 'control', 'status', 'outage', 'FI_Installation'],
                               layer_name = 'Original Intact Distribution Grid',
                               info_name = 'Line Information')


    #Update choropleth with point clusters
    node_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/Blue_icon.png'
    layer_map_1 = add_point_clusters(layer_map_1, 
                                     df_node, 
                                     attr = ['Node', 'lat', 'lon'], 
                                     node_image = node_icon, 
                                     node_name = 'Node', 
                                     info_name = 'Node Information')


    fault_indicators_OFF_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/fi_off.png'
    layer_map_1 = add_point_clusters(layer_map_1, 
                                     df_FI_OFF_location , 
                                     attr = ['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status'],
                                     node_image = fault_indicators_OFF_icon, 
                                     node_name = 'Fault indicators OFF',
                                     icon_size=(15,15), 
                                     info_name = 'Fault Indicator Information')


    # Fullscreen
    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True).add_to(layer_map_1)

    layer_map_1.add_child(folium.LayerControl()) #Add layer control to toggle on/off

    #current_path = os.getcwd()
    #filepath = current_path + '/output/intact_layer_map.html'
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    filepath = os.path.join(data_folder, "intact_layer_map.html")
    layer_map_1.save(filepath)
    print('finish save intact_layer_map.html')

gen_map_1()


### 2nd map
def gen_map_2():
    locations = 41.7557, -72.7831
    bound_box = [[41.7507, -72.7340], [41.7183, -72.6720]]

    #Create basemap specifying map center, zoom level, and using the default OpenStreetMap tiles
    layer_map_2 = folium.Map(location = locations, zoom_start = 11, control_scale = True, tiles='Stamen Terrain')
    layer_map_2.fit_bounds(bound_box)


    #Update basemap with tile Layers
    folium.TileLayer('stamentoner').add_to(layer_map_2)
    folium.TileLayer('stamenwatercolor').add_to(layer_map_2)
    folium.TileLayer('openstreetmap').add_to(layer_map_2)
    folium.TileLayer('cartodbdark_matter').add_to(layer_map_2)
    folium.TileLayer('cartodbpositron').add_to(layer_map_2)


    #Update basemap with Polyline Layer
    layer_map_2 = add_polyline(layer_map_2, 
                               df_merged_line2, 
                               attr = ['Node A', 'Node B', 'Weather Prob', 'Length (ft.)', 'Config.', 'control', 'status', 'outage', 'FI_Installation'],
                               layer_name = 'Failure Estimation using Weather', 
                               prob_name = 'Weather_Prob_color',
                               info_name = 'Line Information')



    #Update choropleth with point clusters
    node_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/Blue_icon.png'
    layer_map_2 = add_point_clusters(layer_map_2, 
                                     df_node, 
                                     attr = ['Node', 'lat', 'lon'], 
                                     node_image = node_icon, 
                                     node_name = 'Node', 
                                     info_name = 'Node Information')


    damaged_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/damage_icon.png'
    layer_map_2 = add_point_clusters(layer_map_2, 
                                     df_damaged_line2, 
                                     attr = ['Node A', 'Node B', 'lat', 'lon'], 
                                     node_image = damaged_icon, 
                                     node_name = 'Actual fault locations',
                                     icon_size=(25,25), 
                                     info_name = 'Outage Information')


    # Fullscreen
    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True).add_to(layer_map_2)

    layer_map_2.add_child(folium.LayerControl()) #Add layer control to toggle on/off

    #current_path = os.getcwd()
    #filepath = current_path + '/output/layer_map_2.html'
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    filepath = os.path.join(data_folder, "layer_map_2.html")
    layer_map_2.save(filepath)
    print('finish save layer_map_2.html')


### 3rd map
def gen_map_3():
    locations = 41.7557, -72.7831
    bound_box = [[41.7507, -72.7340], [41.7183, -72.6720]]

    #Create basemap specifying map center, zoom level, and using the default OpenStreetMap tiles
    layer_map_3 = folium.Map(location = locations, zoom_start = 11, control_scale = True, tiles='Stamen Terrain')
    layer_map_3.fit_bounds(bound_box)


    #Update basemap with tile Layers
    folium.TileLayer('stamentoner').add_to(layer_map_3)
    folium.TileLayer('stamenwatercolor').add_to(layer_map_3)
    folium.TileLayer('openstreetmap').add_to(layer_map_3)
    folium.TileLayer('cartodbdark_matter').add_to(layer_map_3)
    folium.TileLayer('cartodbpositron').add_to(layer_map_3)


    #Update basemap with  Polyline Layers
    layer_map_3 = add_polyline(layer_map_3, 
                               df_merged_line2, 
                               attr = ['Node A', 'Node B', 'FI Prob', 'Length (ft.)', 'Config.', 'control', 'status', 'outage', 'FI_Installation'],
                               layer_name = 'Failure Estimation using FI', 
                               prob_name = 'FI_Prob_color', 
                               info_name = 'Line Information')



    #Update choropleth with point clusters
    node_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/Blue_icon.png'
    layer_map_3 = add_point_clusters(layer_map_3, 
                                     df_node, 
                                     attr = ['Node', 'lat', 'lon'], 
                                     node_image = node_icon, 
                                     node_name = 'Node', 
                                     info_name = 'Node Information')


    damaged_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/damage_icon.png'
    layer_map_3 = add_point_clusters(layer_map_3, 
                                     df_damaged_line2, 
                                     attr = ['Node A', 'Node B', 'lat', 'lon'], 
                                     node_image = damaged_icon, 
                                     node_name = 'Actual fault locations',
                                     icon_size=(25,25), 
                                     info_name = 'Outage Information')


    fault_indicators_OFF_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/fi_off.png'
    layer_map_3 = add_point_clusters(layer_map_3, 
                                     df_FI_OFF_location , 
                                     attr = ['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status'],
                                     node_image = fault_indicators_OFF_icon, 
                                     node_name = 'Fault indicators OFF',
                                     icon_size=(15,15), 
                                     info_name = 'Fault Indicator Information')


    fault_indicators_ON_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/fi_on.png'
    layer_map_3 = add_point_clusters(layer_map_3, 
                                     df_FI_ON_location , 
                                     attr = ['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status'],
                                     node_image = fault_indicators_ON_icon, 
                                     node_name = 'Fault indicators ON',
                                     icon_size=(15,15), 
                                     info_name = 'Fault Indicator Information')


    # Fullscreen
    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True).add_to(layer_map_3)

    layer_map_3.add_child(folium.LayerControl()) #Add layer control to toggle on/off


    #current_path = os.getcwd()
    #filepath = current_path + '/output/layer_map_3.html'
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    filepath = os.path.join(data_folder, "layer_map_3.html")
    layer_map_3.save(filepath)
    print('finish save layer_map_3.html')


### 4th map
def gen_map_4():
    locations = 41.7557, -72.7831
    bound_box = [[41.7507, -72.7340], [41.7183, -72.6720]]

    #Create basemap specifying map center, zoom level, and using the default OpenStreetMap tiles
    layer_map_4 = folium.Map(location = locations, zoom_start = 11, control_scale = True, tiles='Stamen Terrain')
    layer_map_4.fit_bounds(bound_box)


    #Update basemap with tile Layers
    folium.TileLayer('stamentoner').add_to(layer_map_4)
    folium.TileLayer('stamenwatercolor').add_to(layer_map_4)
    folium.TileLayer('openstreetmap').add_to(layer_map_4)
    folium.TileLayer('cartodbdark_matter').add_to(layer_map_4)
    folium.TileLayer('cartodbpositron').add_to(layer_map_4)


    #Update basemap with  Polyline Layers
    layer_map_4 = add_polyline(layer_map_4, 
                               df_merged_line2, 
                               attr = ['Node A', 'Node B', 'FI+Weather Prob', 'Length (ft.)', 'Config.', 'control', 'status', 'outage', 'FI_Installation'],
                               layer_name = 'Failure Estimation using Weather + FI', 
                               prob_name = 'FI+Weather_Prob_color', 
                               info_name = 'Line Information')


    #Update choropleth with point clusters
    node_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/Blue_icon.png'
    layer_map_4 = add_point_clusters(layer_map_4, 
                                     df_node, 
                                     attr = ['Node', 'lat', 'lon'], 
                                     node_image = node_icon, 
                                     node_name = 'Node', 
                                     info_name = 'Node Information')


    damaged_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/damage_icon.png'
    layer_map_4 = add_point_clusters(layer_map_4, 
                                     df_damaged_line2, 
                                     attr = ['Node A', 'Node B', 'lat', 'lon'], 
                                     node_image = damaged_icon, 
                                     node_name = 'Actual fault locations',
                                     icon_size=(25,25), 
                                     info_name = 'Outage Information')


    fault_indicators_OFF_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/fi_off.png'
    layer_map_4 = add_point_clusters(layer_map_4, 
                                     df_FI_OFF_location , 
                                     attr = ['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status'],
                                     node_image = fault_indicators_OFF_icon, 
                                     node_name = 'Fault indicators OFF',
                                     icon_size=(15,15), 
                                     info_name = 'Fault Indicator Information')


    fault_indicators_ON_icon = 'https://raw.githubusercontent.com/dvu4/grid_folium/master/data/fi_on.png'
    layer_map_4 = add_point_clusters(layer_map_4, 
                                     df_FI_ON_location , 
                                     attr = ['Node A', 'Node B', 'lat', 'lon', 'Fault Indicator Status'],
                                     node_image = fault_indicators_ON_icon, 
                                     node_name = 'Fault indicators ON',
                                     icon_size=(15,15), 
                                     info_name = 'Fault Indicator Information')


    # Fullscreen
    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True).add_to(layer_map_4)

    layer_map_4.add_child(folium.LayerControl()) #Add layer control to toggle on/off




    #current_path = os.getcwd()
    #filepath = current_path + '/output/layer_map_4.html'
    current_path = os.getcwd()
    data_folder = os.path.join(current_path,"output")
    filepath = os.path.join(data_folder, "layer_map_4.html")
    layer_map_4.save(filepath)
    print('finish save layer_map_4.html')

'''
def main():
    #df_node = read_node_data()
    #df_line = read_line_data(df_node)
    #df_failure_estimation, df_FI_damaged_line, df_Weather_damaged_line, df_FI_Weather_damaged_line = read_fault_data(df_node)
    #df_merged_line2, df_FI_OFF_location, df_FI_ON_location = read_fault_indicator_data(df_node, df_line, df_failure_estimation)
    gen_map_1()
    gen_map_2()
    gen_map_3()
    gen_map_4()   
'''  
    
#if __name__ == "__main__":
    #main()
