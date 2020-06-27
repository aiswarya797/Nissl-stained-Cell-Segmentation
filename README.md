# Nissl-stained-Cell-Segmentation
This repository contains all the code written during my thesis at CCBR, IIT Madras. 

## Command to run the code from command line :
    python cell_segment_basic.py --inputpath '/home/aiswarya/data_test/input' --outputpath '/home/aiswarya/data_test/output' --targetPresent True --jsonPath '/home/aiswarya/CCBR-IITM-Thesis/data.json'

Here. --inputpath contains the path to the input files, --outputpath contains path to output files, wherein the segmented output images are saved, --targetPresent is a boolean, which states if target contours are available (for example, if human annotated contours are available in json format, this variable is set to True)
--jsonPath is the path to the json file, if targetPresent is true. Otherwise, this variable can be neglected.
