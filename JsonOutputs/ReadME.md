###### The inputs folder should have the data points. If the corresponding human annotated contours are available in json format, this can be included into the 'input_json' folder. The path to the input data json file is given as input argument to --jsonInputPath. Here, data1.json is empty. Update this file, if human annotated json file available.

###### Format of the json file: 
* Key : "External ID"  Value : The file name
* Key : "Label" Value : Dictionary of contours. 
** Kindly check the output.json file to understand the format **
    
###### The outputs folder will have the data points with the algorithm predicted contours. If the contours need to be saved in json format, the path to the output json file is given as input argument to --jsonOutputPath and the json file shall be saved in the respective folder.
