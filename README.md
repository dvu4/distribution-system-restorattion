# dist-restoration


![alt text](https://github.com/dvu4/distribution-system-restorattion/blob/a07885d9d1b2dcf0f48c7dd7637080b375842a65/images/logo-grid-mod-lc.png)
### Folder Structure 
```
.
|	└── fault_indicator_data.xls	
└──data  								# input data for fault estimation and map generation
|	└── line data_py.xls
|	└── node data_123node.xls
|	└── node data_py.xls
|	└── weather_data.xls
|	└── Flood_damge_123.xlsx
|	└── Flood_damge_128500.xlsx
|	└── Flood_weather_metric.xlsx
|	└── Winter_Storm_Ice_weather_metric.xlsx
|	└── Winter_Storm_Wind_weather_metric.xlsx
|	└── Winter_Storm_damge_123.xlsx
|	└── Winter_Storm_damge_128500.xlsx
|	└── test_system_data
|					└── ckt12_ieee8500_system_data.dat
| 					└── IEEE123_system_data.dat
|
└──ui_forms 							# ui library
|	└──Window_1.ui						# ui for data import module 
|	└──Window_2.ui						# ui for fault estimation module 
|	└──Window_3.ui						# ui for crew dispatch module
|	└──Window_4.ui.   					# ui for service restoration module 
|	└──MainWindow.ui. 					# ui for main modules
|	
└──xlsx_data							# input data for power grid 
└──images
└──docs
└──notebooks
└──output								# output data 
└──fault_estimation_module.py.   		# Extracting the fault location , indicator location and weather-related damage probability 
└──fault_location_import_module.py.    	# Reading the fault location 
└──map_module.py. 						# Visualizing fault estimation with Folium 
└──flood_map_module.py 	 				# Visializing flood-related damage probability
└──storm_map_module.py 					# Visializing storm-related damage probability
└──MainWindow.py			
└──Window_1.py	  						# Importing system data, choose setting files before excuting the module 2, 3 and 4		
└──Window_2.py	  						# Generating and displaying the map for each type of fault  
└──Window_3.py							# Script for crew dispatch module
└──Window_4.py							# Script for solving DSR problem
└──restoration_module.py				# Formulating and solving the restoration problem ().
└──visualization_module.py. 			# Visualizing the solution at each step, load demand, reading solution files.  
└──plot_dss_module						# Plotting the the topology for DSS system
└──data_import_module.py 				# Combining all setting files (node, capacitor, ...) into system file (.dat).
└──README.md
