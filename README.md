# Nissl-stained-Cell-Segmentation
This repository contains all the code written during my thesis at CCBR, IIT Madras. 

## Command to run the code from command line :
    python cell_segment_basic.py --inputpath '/home/aiswarya/data_test/input_images' --outputpath '/home/aiswarya/data_test/output' --targetPresent False --jsonInputPath '/home/aiswarya/data_test/input_json/data1.json' --saveJson True --jsonOutPath '/home/aiswarya/data_test/output_json/output.json'

Here. --inputpath contains the path to the input files, --outputpath contains path to output files, wherein the segmented output images are saved, --targetPresent is a boolean, which states if target contours are available (for example, if human annotated contours are available in json format, this variable is set to True)
--jsonInputPath is the path to the json file, if targetPresent is true. Otherwise, this variable can be neglected.  --jsonOutputPath is the path to the folder where in the predicted contours shall be saved in json format, if saveJson is True. Otherwise this variable is also neglected

## cell_segment_basic.ipynb is a colab file
This file can be uploaded to a colaboratory file on Google Drive. Make sure that the folders 'Input_Images', 'input_json', 'output_images' and 'output_json' are present (as present in JsonOutputs). These file paths must be provided inside the 'main' function.
