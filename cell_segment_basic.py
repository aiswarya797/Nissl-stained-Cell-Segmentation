# TODO (under progress)-> Make OOP based code. 
## Store all images in a folder in 'path' variable
# Tackles the following issues : 1) Cell contours
#								 2) Clump split
#								 3) Cluster the layers on basis of contour area


########################################################################################################################################						

						####     NISSL CELL SEGMENTATION ALGORITHM     ####

########################################################################################################################################						

## Dependencies

import json
import argparse
import sys
import cv2
import numpy as np
from scipy import stats
from operator import itemgetter
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from sklearn.cluster import KMeans, MiniBatchKMeans,AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
#import imutils
import scipy.stats as ss
from scipy.stats import ks_2samp, pearsonr
import seaborn as sns
from mpl_toolkits import mplot3d
import json
import os
import math
import cv2


## Input Images Folder

def get_image_inputs(path):
	images = os.listdir(path)
	print(images)
	image_files_list = []
	i = 0
	for image in images:
	  im = path + '/' + image
	  i = i+1
	  image_files_list.append(im)
	print('image file')
	return image_files_list

#path = '/home/aiswarya/data_test'
#image_files_list = get_image_inputs(path)
#print(image_files_list)
## Util Functions

def make_dict(contour_points):
	mapping = {}
	for i in range(len(contour_points)):
		value = i
		key = tuple(np.asarray(contour_points[i][0]))
		mapping[key] = value
	
	return mapping


def getDict(contours, values):
	mapping = {}
	for i in range(len(contours)):
		cnt = contours[i]
		mapping[tuple(cnt.ravel())] = values[i]
		
	return mapping


#Step 1: Image Pre-processing
#* Convert the image to grayscale.    
#* Bitwise NOT operation is performed on Image: Inversion of input array elements. This improves contrast in the image.
#* Threshold the image using OTSU Thresholding
#(this will make the image with background white and objects black).

def contraster(img_file):
  #-----Reading the image-----------------------------------------------------
  img = cv2.imread(img_file, 1)
  #cv2.imshow("img",img) 

  #-----Converting image to LAB Color model----------------------------------- 
  lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

  #-----Splitting the LAB image to different channels-------------------------
  l, a, b = cv2.split(lab)

  #-----Applying CLAHE to L-channel-------------------------------------------
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  cl = clahe.apply(l)

  #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
  limg = cv2.merge((cl,a,b))

  #-----Converting image from LAB Color model to RGB model to GRAY model--------------------
  final = cv2.cvtColor(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2GRAY)
  #cv2.imshow('final', final)
  return final
  #_____END_____#
  
## 	find_basic_concave_points function finds the elements 
def find_basic_concave_points(contour,n,hull,m): #n>m ## Assumption #elements in contour not in hull
	arr = []
	diff = []
	contour1 = contour.tolist()
	hull1 = hull.tolist()
	hul = hull1
	for ele1 in contour1:
		flag = True
		for ele2 in hull1:
			if ele1==ele2:
				arr+=[ele1]
				hul.remove(ele2)
				flag = False
				break
			
		if (flag):
			diff+=[ele1]
	
	return diff

## norm == True does not work well
# TODO find why

## angle_concave function finds the angle subtended at each concave point of each contour.
def angle_concave(contour, concave_point, parameters):
  norm = parameters['NORM']
  if norm == False:
    mapping = make_dict(contour)
    index = mapping[tuple(concave_point)]
    prev = index-1
    next_ = index+1
    if index==0:
      prev = len(contour)-1
    if index == len(contour)-1:
      next_ = 0
    a = contour[prev][0]
    a = np.array(a.tolist())
    b = np.array(concave_point)
    c = contour[next_][0]
    c = np.array(c.tolist())
      
    ba = a - b
    bc = c - b
      
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)	

  else:
    mapping = make_dict(contour)
    index = mapping[tuple(concave_point)]
    prev = index-1
    next_ = index+1
    if index==0:
      prev = len(contour)-1
    if index == len(contour)-1:
      next_ = 0


    a = contour[prev][0]
    a = np.array(a.tolist())
    b = np.array(concave_point)
    c = contour[next_][0]
    c = np.array(c.tolist())
      
    _BA_ = np.sqrt(np.square(b-a))
    _BC_ = np.sqrt(np.square(c-b))

    norm_BA = np.linalg.norm(_BA_)
    norm_BC = np.linalg.norm(_BC_)

    cos_ang = np.dot(_BA_, _BC_)/(norm_BA*norm_BC)
    angle = np.arccos(cos_ang)

  return np.degrees(angle)

## find_mid_pt function finds the mid point of the line joined by two points  pt1 and pt2
def find_mid_pt(pt1, pt2):
  pt1_x = pt1[0][0]
  pt1_y = pt1[0][1]

  pt2_x = pt2[0][0]
  pt2_y = pt2[0][1]

  mid_pt_x = (pt1_x+pt2_x)/2
  mid_pt_y = (pt1_y+pt2_y)/2

  return (mid_pt_x, mid_pt_y)

def euclidean(x1,y1,x2,y2):
	# print('x1', x1)
	# print('y1', y1)
	# print('x2', x2)
	# print('y2', y2)
	return np.sqrt(np.add(np.square(np.subtract(x1,x2)),np.square(np.subtract(y1,y2))))
	
## nearest_pair_point function finds the nearest point to coord1 from a set of points
def nearest_pair_point(coord1, set_of_points):
	minimum = 100000
	x1 = coord1[0][0]
	y1 = coord1[0][1]
	for coord2 in set_of_points:
		x2 = coord2[0]
		y2 = coord2[1]
		euclideandis = euclidean(x1,y1,x2,y2)
		if euclideandis < minimum:
			minimum = euclideandis
			min_point = [x2,y2]
			
	return min_point

def where_index(arr, ele):
	index = -1
	for i in range(len(arr)):
		a = arr[i]
		if a[0] == ele[0] and a[1]==ele[1]:
			index = i
			break

	return index
	

"""---
**Get the Concave Points**

* *Concave points are those points where two clumps meet*

---

---
OLD VERSION

---
"""

def find_concave_points_using_hull(contour,l1, hull, l2, parameters,concavePT_angle_dict):
# https://stackoverflow.com/questions/29477049/opencv-find-concave-hull
	# l1>=l2
	#num_of_concave_points = np.abs(l2,l1)
  ANGLE_THRESHOLD = parameters['ANGLE_THRESHOLD']
  cnt = []
  hul = []
  angles = []
  for i in range(len(contour)):
    ele = contour[i]
    ele = ele.ravel()
    cnt.append(ele)
    
  for i in range(len(hull)):
    ele = hull[i]
    ele = ele.ravel()
    hul.append(ele)
    
  cnt = np.asarray(cnt)
  hul = np.asarray(hul)
  arr = find_basic_concave_points(cnt,l1, hul,l2)  # TODO Make this more efficient

  concave_points = []
  for ele in arr:
    angle = angle_concave(contour, ele,parameters)
    if angle < ANGLE_THRESHOLD:
      concave_points.append(ele)
      angles.append(angle)
      concavePT_angle_dict[tuple(ele)] = angle
    
  #print(arr)
  #if arr:  #Means list is not empty
  return concave_points, angles,concavePT_angle_dict

"""---
NEW VERSION

---
"""

def find_concave_points_using_curvature(contour,concavePT_angle_dict):
  angles = []
  length = len(contour)
  count = 0
  concave_points_found = []
  for index in range(length):
    if index ==0:
      last_point = contour[length-1]
    else:
      last_point = contour[index-1]
    if index == length-1:
      next_point = contour[0]
    else:
      next_point = contour[index+1]

    curr_point = contour[index]
    midPoint = find_mid_pt(last_point,next_point)
#https://stackoverflow.com/questions/50670326/how-to-check-if-point-is-placed-inside-contour/50670359
    dist = cv2.pointPolygonTest(contour,midPoint,True)
    if dist<0:
      if count == 0:
        concave_points_found = curr_point
        count+=1
      else:
        concave_points_found += curr_point
        count+=1

  print(count)
  for ele in concave_points_found:
    angle = angle_concave(contour, ele,parameters)
    angles.append(angle)
    concavePT_angle_dict[tuple(ele)] = angle

  return concave_points_found,angles,concavePT_angle_dict

## Wrapper Function
def find_concave_points(contour, l1, hull, l2, parameters, concavePT_angle_dict):
	func = parameters['conptFunc']
	if func== 'find_concave_points_using_hull':
		concave_points_found,angles,concavePT_angle_dict  = find_concave_points_using_hull(contour,l1, hull, l2, parameters,concavePT_angle_dict)
	elif func == 'find_concave_points_using_curvature':
		concave_points_found,angles,concavePT_angle_dict  = find_concave_points_using_curvature(contour,concavePT_angle_dict)
	
	return concave_points_found,angles,concavePT_angle_dict

"""---
Pair Finder : This function finds the pair to concave point **X**,  in case there is not pair in the list of concave points but the angle subtended at **X** is small enough.

---
"""

## This is the pair searcher, when pair not found for a concave point, and we take diametrically opposite element as its pair.
## TODO This logic has the issue that, the line can go outside of the contour. Also, diametrically opposite element need not be always the mid point of the remaining contour. 
def single_point(contour, index):
  arr = np.zeros((len(contour),2))
  b = True
  ind = index+1
  a =0 
  while b:
    if ind == len(contour):
      ind = 0
    if ind == index:
      b = False
      continue
    arr[a] = contour[ind]
    a = a+1
    ind = ind+1

  if len(arr)%2 == 1:
    return arr[math.floor(len(arr)/2)]

  else:
    S = []
    S.append(arr[math.floor(len(arr)/2-1)])
    S.append(arr[math.floor(len(arr)/2)])
    ele = contour[ind]
    return nearest_pair_point(ele, S)

"""---
Given set of concave points. Find pair for each of them from the set of concave points.

We use variable sized rectangle in order to see which point (potential pair) falls inside the rectangle.

---
"""

## Variable Sized Rectangle
## this function says if any concave point is inside the rectangle
def rectContains(rect,points_set):	#rect = (a,b,c,d) #a,b are the top-left coordinate of the rectangle and (c,d) be its width and height.
#	print('inside rectContains')
	pnt = points_set[0]
	logic = False
	
	for pt in points_set:
		logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
		if logic:
			pnt = pt
			logic = logic
			break
	return pnt,logic

## point_on_contour function finds that point on the contour, at the specified index.
def point_on_contour(current_index, direction, contour):

	if direction == 'clw':
		if current_index == len(contour)-2:
			ind = 0
		elif current_index == len(contour)-1:
			ind = 1
		else:
			ind = current_index+2
	
	if direction == 'aclw':
		if current_index == 1:
			ind = len(contour)-1
		elif current_index ==0:
			ind = len(contour)-2
		else:
			ind = current_index-2

	ele = contour[ind]
	return ele


## point_on_hull function finds that point on the hull, at the specified index.
def point_on_hull(current_index, direction, contour, hull):
	
	b = True
	
	if direction == 'clw':
		#ind = current_index+2
		ind = current_index+1
		while b:
			if ind >= len(contour):
				#ind = ind%len(contour)
				ind = 0
			
			if contour[ind] in hull:
				b = False
				ele = contour[ind]
				return ele
				
			ind = ind+1
			
	if direction == 'aclw':
		#ind = current_index-2
		ind = current_index-1
		while b:
			if ind < 0:
				#ind = ind*-1
				#ind = len(contour)-ind%len(contour)
				ind = len(contour)-1
			
			if contour[ind] in hull:
				b = False
				ele = contour[ind]
				return ele
				
			ind = ind-1			
	
## 	initial_rectangle function gets the initial rectangle around the chosen concave point.
def initial_rectangle(ele, hull, contour, mapping, parameters):
	
	#print('here - IR')
	n = parameters['RECT_OFFSET']
	# print('ele', ele)
	ele = tuple(ele)
	index = mapping[ele]
	
	if len(contour)<5:
		point_hull1 = point_on_hull(index, 'clw', contour, hull)
		point_hull2 = point_on_hull(index, 'aclw',contour,hull)
	
	else:
		point_hull1 = point_on_contour(index, 'clw', contour)
		point_hull2 = point_on_contour(index, 'aclw',contour)

	#print('found hull')
	offset = euclidean(point_hull1[0][0],point_hull1[0][1],point_hull2[0][0],point_hull2[0][1])/n
	width = euclidean(point_hull1[0][0],point_hull1[0][1],point_hull2[0][0],point_hull2[0][1])
	height = width

	rect = [point_hull1[0][0], point_hull1[0][1], width, height]
	return rect, offset

## bounding_rectangle function finds the variable sized rectangle, which grows till the time, either the concave pair is found or the maximum number of iterations threshold is passed.	
def bounding_rectangle(concave_point, concave_points, hull, contour, mapping,parameters):
	
	#print('here - BR')
	rect, offset = initial_rectangle(concave_point, hull, contour, mapping,parameters)
	#print('returned from IR')
	flag = True
	iterations = 0
	pair = []
	while flag:
		pt,logic = rectContains(rect, concave_points)
		#print('got logic')
		if logic == True:
			flag = False
			pair.append(pt)
		
		elif iterations == 15:
			pair.append([])	# As otherwise, the return dimensions do not tally. 
			flag = False
				
		else:
			rect[3] = rect[3] + offset
			iterations = iterations + 1
		
	return pair	# 2D array returned

## This is not greedy way of finding pairs. We find all possible pairs and see which is most appropriate one.
## Most appropriate is the pair at least distance
## Applying single_point() function always is not correct. There are many cases when a random point becomes a concave point and 
## it has not pair to be matched with. A random point will not subtend an acute angle.
## So we find the angle subtended by a point (if no pair was found). If angle<90, only then shall we apply single_point(). 
## Otherwise the point is discarded.
def find_pairs_optimal(concave_points, contour, hull,concavePT_angle_dict):
	mapping = make_dict(contour)
	# pairs = []
	concave_points = np.array(concave_points)
	# print('concave_points', concave_points)
	length = len(concave_points)
	pairs_dict = {}
	#concavePT_angle_dict = concavePT_angle_image_dict[image]
	for i in range(length):
		key = tuple(concave_points[i])
		temp_dict = {}
		for j in range(length):
				pair = bounding_rectangle(concave_points[i],[concave_points[j]], hull,contour,mapping,parameters)
				angle = concavePT_angle_dict[key]
				# print(pair[0])
				temp_dict[tuple(pair[0])] = angle
				
		pairs_dict[key] = temp_dict

	optimal_pairs= []
	flag_global = True
	for i in range(length):
		key = tuple(concave_points[i])
		temp_dict = pairs_dict[key]
		pairs = list(temp_dict.keys())
		angles = list(temp_dict.values())

		for k in range(len(pairs)):
			if len(pairs[k])==0:
				ang = angles[k]
				if ang<=90:
					index = mapping[tuple(concave_points[i])]
					pairs[k] = single_point(contour, index)


		min_dis = 1000000
		flag = False
		for k in range(len(pairs)):
			if len(list(pairs[k])) == 0:
				continue
			else:
				dis = euclidean(concave_points[i][0],concave_points[i][1],pairs[k][0],pairs[k][1])
				if dis<min_dis:
					min_dis = dis		
					req_pair = list(pairs[k])
					flag = True
					flag_global = False

		if flag:
			optimal_pairs.append([concave_points[i], req_pair])
	 

	return optimal_pairs, flag_global

## This is a greedy pair finder.
## Once a pair is found, they are removed from concave points list
def find_pairs_greedy(concave_points, contour, hull):
	
	mapping = make_dict(contour)
	pairs = []
	a = 0
	flag = True
	flag_global = True
	concave_points = np.array(concave_points)
	print('concave_points', concave_points)
	latest_len = len(concave_points)
	while flag:
		if len(concave_points)==0 or len(concave_points)==1 or a>=latest_len:
				flag = False
				break
		print('a', a)
		concave_point = concave_points[a]
		print('concave_point', concave_point)
		pair = bounding_rectangle(concave_point,concave_points, hull,contour,mapping,parameters)
		print('pair', pair)

		#if index_concave>-1:
		#	concave_points = np.delete(concave_points,index_concave, axis=0)

		if len(pair[0]) == 0:
				print('pair was not there')
				print(concave_points)
				a = a+1
				continue

		if pair[0][0]==concave_point[0] and pair[0][1]==concave_point[1]:
			pair[0] = []

		if len(pair[0]) == 0:
				print('nothing')
				a = a+1		

		else:
			pt = pair[0]
			print('concave_points ',concave_points, 'pair', pt )
			index_pair = where_index(concave_points,pt)
			concave_points = np.delete(concave_points,index_pair, axis=0)
			index_concave = where_index(concave_points,concave_point)
			concave_points = np.delete(concave_points,index_concave, axis=0)
	 
		if len(concave_points)==0 or len(concave_points)==1 or a>=len(concave_points):
					flag = False
		
		latest_len = len(concave_points)
		pairs.append([concave_point, pair[0]])
		flag_global = False
		print('concave_points', concave_points)

		
		
	return pairs, flag_global

# Wrapper for finding pairs
# Loop through all the contours, if concave points are not present, the contour shall be added to the “non_clumped_contours" set.
# A variable sized rectangle is used that will grow lengthwise, hoping to include the corresponding concave point pair inside the rectangle at some point of time.
#An initial rectangle is formed around the selected concave point as follows:  The adjacent points to the concave point on the convex hull on either side form one side of the rectangle (breadth). The length is initialised as the breadth itself. An offset, which is the amount of increase in the length at each iteration is specified as breadth/n, here n=1.
#Then for at least 10 iterations, I keep increasing the rectangle length and check if any other concave point is present inside the rectangle. If not, then find the point on the contour that is diametrically opposite and nearest to the current point.
#A concave point may get assigned to multiple pair points. In that case, the point with least euclidean distance shall be assigned as the pair.
#Even after rejecting many non-candidate pairs, there can be still pairs, which need not to be joined. They are removed in the next step.

def find_pairs(concave_points, contour, hull,concavePT_angle_dict, parameters):
  func = parameters['pair_finder']
  if func == 'optimal':
    pairs, flag_global = find_pairs_optimal(concave_points, contour, hull,concavePT_angle_dict)
  elif func == 'greedy':
    pairs, flag_global = find_pairs_greedy(concave_points, contour, hull)

  return pairs, flag_global

"""---
* Idea of Max Min

  * Each line is taken. It is drawn and the area of division is taken, this is done for every line.
  * Out of all the lines that line is chosen which gives maximum minimum area.
  * The smaller area is taken as first contour. 
  *In remaining contours, remaining lines are taken and same procedure is followed.

---
"""

## Max - Min area algorithm. 
## Idea of Max Min (FOR EACH CONTOUR)
## Each line is taken. It is drawn and the minimum area of the division is taken, for every line and this value is noted. Out of all the lines, that line (which is the optimal line) is chosen which gives the maximum minimum area. The smaller area (formed by this optimal line) is taken as the first contour. In the remaining part of the contour, remaining lines are taken and the same procedure is followed and stops within four iterations. 

def max_min_area_cut(contour, lines):
	mapping = make_dict(contour)
	max_area_cuts = 0
	req_line = [] #initialize
	req_line_ind = 0
	return_con = contour
	remain_con = contour
	start = 0
	end = len(contour)
	ind = 0
	
	b = True
	
	for line in lines:	#line = [point1, point2]
		point1 = line[0]
		point2 = line[1]	
		try:	
			ind1 = mapping[tuple(point1)]
			ind2 = mapping[tuple(point2)]
			b = False
		except:
			continue
		
		temp = 0
		if ind2<ind1:
			temp = ind1
			ind1 = ind2
			ind2 = temp
		
		# addnl_points = find_mid_pt(contour[ind1], contour[ind2])
		
		con1 = contour[start:(ind1+1)].tolist()
		# con1 += addnl_points
		con1 += contour[ind2:end].tolist()
		
		con2 = contour[ind1:(ind2+1)].tolist()
		# con2 += addnl_points
		
		# ERROR w/o lines 563 and 564 : Expected Ptr<cv::UMat> for argument '%s'
		con1 = np.array(con1).astype(np.float32)
		con2 = np.array(con2).astype(np.float32)
		
		area1 = cv2.contourArea(con1)
		area2 = cv2.contourArea(con2)
	 	
		if area1<area2:
			if max_area_cuts < area1:
				max_area_cuts = area1
				req_line = line
				req_line_ind = ind
				return_con = con1
				remain_con = con2
				
		else:
			if max_area_cuts < area2:
				max_area_cuts = area2
				req_line = line
				req_line_ind = ind
				return_con = con2
				remain_con = con1
		ind = ind+1

	return max_area_cuts, req_line, req_line_ind, return_con, remain_con,b

## This function finds the clump splits
## Split line algorithm
## There are many lines that can be drawn inside the contour (each line joins the concave point pairs). Some might intersect as well. A method is needed to find the required lines and exclude all the other lines. 

## For each contour:
##        1. Loop through all the lines and see which gives the max min area (Explained Below). 
##        2. Once that line is obtained, I remove that part of the contour and (1) is repeated on the remaining set of lines, for the remaining part of the contour.
## If no pair is present, the angle subtended by the contour at the concave point is found. If it is acute (<=90), then a straight line joining the point with a point on the other side of contour is drawn, in the direction of concavity. Otherwise, the concave point is discarded.

def retrieve_contour(contour, lines):

	ite = 0
	final_set = []
	final_lines = []
	#max_area_cut = 0
	return_con = []
	remain_con = []
	#lines.sort()
		
	while ite<10 and len(lines)>0:
		max_area_cut, req_line,req_line_ind, return_con, remain_con,b = max_min_area_cut(contour, lines)
		if not b:
			final_lines.append(req_line)
			final_set.append(return_con)
			lines = np.delete(lines, req_line_ind)
			contour = remain_con
			ite = ite+1
		else:
			final_set.append(contour)	
			ite = 20
		
	final_set.append(remain_con)
	return final_set, final_lines

## This function makes only one cut per clump -> NOT USED.	
def one_cut(contour, lines):
	final_set = []
	final_lines = []
	
	start = 0
	end = len(contour)
	
	mapping = make_dict(contour)
	
	#lines.sort()
	line = lines[0]
	final_lines.append(line)
	
	point1 = line[0]
	point2 = line[1]
	
	ind1 = mapping[tuple(point1)]
	ind2 = mapping[tuple(point2)]
	
	temp = 0
	if ind2<ind1:
		temp = ind1
		ind1 = ind2
		ind2 = temp
	
	con1 = contour[start:(ind1+1)].tolist()
	con1 += contour[ind2:end].tolist()
	con2 = contour[ind1:(ind2+1)].tolist()	
	
	final_set.append(con1)
	final_set.append(con2)
	
	return final_set, final_lines
	
def find_final_contours(contours, line_dict):
# line_dict is a dictionary, key = str(contour), and values are the lines in the contour drawn. Max size 5.	

	final_set_of_contours = []
	final_set_of_lines = []
		
	for contour in contours:
		#try:
		#print(tuple(contour.ravel()))
		lines = line_dict[tuple(contour.ravel())]
		final_set, final_lines = retrieve_contour(contour, lines)
		#final_set, final_lines = one_cut(contour, lines)
		"""
		except:
			print('h')
			final_set = contour.tolist()
			final_lines = []
		"""
		final_set_of_contours.append(final_set)
		final_set_of_lines += final_lines
		
	return final_set_of_contours, final_set_of_lines

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
# Suzuki, S. and Abe, K., Topological Structural Analysis of Digitized Binary Images by Border Following. CVGIP 30 1, pp 32-46 (1985)
# => https://stackoverflow.com/questions/10427474/what-is-the-algorithm-that-opencv-uses-for-finding-contours

## Many contours are just aberrations in background and not really contours, these are very small cells or the border of the rectangular tile, both which needs to be deleted..
def reject_outliers(areas, peris,cntrs, m=2):
	
	areas = np.array(areas).astype(int)
	peris = np.array(peris).astype(int)
	#print('areas',len(areas))
	#print('peris',len(peris))
	mean = np.mean(peris)
	#print('area mean', np.mean(areas))
	#print('peri mean', mean)
	new_areas = []
	new_peris = []
	new_cntrs = []
	j = 0
	for i in range(len(peris)):
		if (abs(peris[i] - mean)< m * np.std(peris)) and peris[i]>6:
			new_areas.append(areas[i])
			new_peris.append(peris[i])
			new_cntrs.append(cntrs[i])
			j = j+1

	#print('new areas', len(new_areas))	
	#print('new peris', len(new_peris))
	#print('new area mean', np.mean(new_areas))
	#print('new peri mean', np.mean(new_peris))
	return new_areas, new_peris, new_cntrs

def find_area_perimeter_hist(final_set_of_subcontours):
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
	areas = []
	peris = []
	cntrs = []
	print(len(final_set_of_subcontours))
	for contour in final_set_of_subcontours:
		area = cv2.contourArea(contour)
		peri = cv2.arcLength(contour, True)
		areas.append(area)
		peris.append(peri)
		cntrs.append(contour)

	print(len(areas))	
	areas, peris,cntrs = reject_outliers(areas, peris, cntrs)
	#https://stackoverflow.com/questions/50756776/how-do-you-select-a-custom-bin-size-for-a-2d-histogram
	xMin = min(areas)
	xMax = max(areas)
	yMin = min(peris)
	yMax = max(peris)
	
	binWidth = 5.0
	binLength = 10.0
	
	print(stats.ttest_rel(areas, peris))

	
	return cntrs, areas, peris

parameters = {'ANGLE_THRESHOLD':100,'NORM':False,'RECT_OFFSET':1, 'conptFunc':'find_concave_points_using_hull','pair_finder':'optimal'}

"""---
Contour raw function finds the contours.

**Inputs** : image_files_list

**Outputs**: 

contours_list

areas_list

peris_list 

final_subcontours

final_areas

final_peris

image_stack

images_predContours_dict

concave_points_dict

contour_splitlines_dict

contour_splitlines_img_dict

angles__contours_image_dict

concaves__contours_image_dict

img_cv2_hull_dict

img_cv2_contours_dict

concavePT_angle_image_dict
"""

## This function is needed as many points on the contour are redundant - they form straight lines in the contour shape. These straight lines can be characterised by the end points and the points in between can be discarded.
def truncate_contours(contour):
  truncated_contours = []
  flag = True
  i = 0
  while flag:
    if i+1 >= len(contour)-1:
      flag = False
      break
    if contour[i][0][0] == contour[i+1][0][0]:
      while (i+1)<len(contour)-1 and contour[i][0][0] == contour[i+1][0][0]:
        i+=1
    elif contour[i][0][1] == contour[i+1][0][1]:
      while (i+1)<len(contour)-1 and contour[i][0][1] == contour[i+1][0][1]:
        i+=1

    
    truncated_contours.append(np.array(contour[i]))
    i+=1

  if i<=len(contour)-1:
    truncated_contours.append(np.array(contour[i]))
  return np.array(truncated_contours)

## This funtion finds all the contours. 
def contour_raw(image_files_list):
	
  final_subcontours = []
  contours_list = []
  areas_list = []
  peris_list = []
  image_stack = []
  final_areas= []
  final_peris = []
  img_cv2_contours_dict = {}
  img_cv2_hull_dict = {}
  images_predContours_dict = {}
  counter = 0
  concaves__contours_image_dict = {}
  contour_splitlines_img_dict = {}
  contour_splitlines_dict = {}
  angles__contours_image_dict = {}
  concave_points_dict = {}
  rects_image_dict = {}
  concavePT_angle_image_dict = {}
  for im in range(len(image_files_list)):
    img = contraster(image_files_list[im])
    concave_points_list = []
    splitlines = []
    rects = []
    print(counter)
    counter = counter + 1
    imgpath = image_files_list[im]
    file_ = imgpath.split('/')[-1]
    #img = cv2.imread(imgpath, 0)
    #cv2.imshow('image', img)
    image= img
    hulls = []
    img = cv2.bitwise_not(img)
    # thresh =  cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    _, thresh =  cv2.threshold(img,20,255,cv2.THRESH_OTSU)
    #_,thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY)	#Assumes the cells are darker than background
    image_stack.append(img)
      
    _,contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # print('num of contours cv2', len(contours))
    contours, areas, peris = find_area_perimeter_hist(contours)
    img_cv2_contours_dict[image_files_list[im]] = contours
    #cv2.drawContours(image, contours, -1, 255, 1) #, cv2.LINE_AA, maxLevel=1)
    #cv2.imshow('Contours', image)
    
    ## Total number of contours 
    #https://stackoverflow.com/questions/13905499/count-contours-in-python-in-opencv  => When using cv instead of cv2	
    
    # For each contour, find the convex hull and draw it
    # on the original image.
    #https://theailearner.com/tag/contours-opencv/
    count =0
    line_dict = {}
    contour_list = []
    contour_not_considered = []
    angles_dict = {}
    concaves_dict = {}
    hulls = []
    concavePT_angle_dict = {}
    for i in range(len(contours)):
      contour = contours[i]
      contour = truncate_contours(contour)
      l1 = len(contour)
      hull = cv2.convexHull(contour)
      hulls.append(hull)
      l2 = len(hull)			
      #cv2.drawContours(img, [hull], -1, 0, 1)
      
      #if i == 4147:
      # concave_points,angles,concavePT_angle_dict = concave_points_finder(contours[i],concavePT_angle_dict)
      concave_points,angles,concavePT_angle_dict = find_concave_points(contour,l1, hull, l2, parameters,concavePT_angle_dict)
      angles_dict[tuple(contour.ravel())] = angles
      concaves_dict[tuple(contour.ravel())] = concave_points
      con_ang_dict = {}
      for p in range(len(concave_points)):
        con_ang_dict[tuple(concave_points[p])] = angles[p]
      if count ==0:
        # print('shdf',concave_points)
        concave_points_list = concave_points
        concave_points_list = list(concave_points_list)
        count+=1
      else:
        # print('sdba', concave_points)
        # print('jhfd', concave_points_list)
        concave_points_list.append(concave_points)
        count+=1
      
      if len(concave_points) == 0:
        contour_not_considered.append(contour)
      else:
        pairs, flag_global = find_pairs(concave_points, contour, hull,con_ang_dict, parameters)  
        # if flag_global:
        #   contour_not_considered.append(contour)
        # else:
        #print('done')
        linelengths = []
        for pair in pairs:
          if pair[1] == []:
            linelength =0
          else:
            x1 = int(pair[0][0])
            y1 = int(pair[0][1])
            x2 = int(pair[1][0])
            y2 = int(pair[1][1])
            linelength = euclidean(x1,y1,x2,y2)
            if linelength !=0 and linelength<30:	# Then not same point #and linelength>4 
              lin = [(x1,y1),(x2,y2)]
              linelengths.append([lin,linelength])
        
        linelengths = sorted(linelengths, key= itemgetter(1))
        if len(linelengths) !=0:
          linelengths = np.asarray(linelengths)
          lines_drawn = linelengths[:,0]
          #print(contour_list[i].ravel().tobytes())
          line_dict[tuple(contour.ravel())] = lines_drawn
          contour_list.append(contour)
          splitlines.append(lines_drawn)
      
    
    concavePT_angle_image_dict[image_files_list[im]] = concavePT_angle_dict
    contour_splitlines_dict[image_files_list[im]] = splitlines
    concaves__contours_image_dict[image_files_list[im]] = concaves_dict
    contour_splitlines_img_dict[image_files_list[im]] = line_dict
    img_cv2_hull_dict[image_files_list[im]] = hulls
    angles__contours_image_dict[image_files_list[im]] = angles_dict
    concave_points_dict[image_files_list[im]] = concave_points_list
    final_set_of_contours, final_set_of_lines = find_final_contours(contour_list, line_dict)
    # print('num of final_set_', len(final_set_of_contours))
    final_set_of_subcontours = []
    
    for i in range(len(final_set_of_contours)):
      f = final_set_of_contours[i][0]
      f = np.asarray(f).astype(np.int32)
      final_set_of_subcontours.append(f)
      
      if len(final_set_of_contours[i]) > 1:
        for j in range(len(final_set_of_contours[i])):
          f = final_set_of_contours[i][j]
          f = np.asarray(f).astype(np.int32)
          final_set_of_subcontours.append(f)

    # print('contours not considered',len(contour_not_considered))   
    final_set_of_subcontours += contour_not_considered
    # print('final_set_of_subcontours', len(final_set_of_subcontours))
    print(len(final_set_of_subcontours))
    
    #final_set_of_subcontours = reject_outliers(final_set_of_subcontours)
    final_set_of_subcontours, areas, peris = find_area_perimeter_hist(final_set_of_subcontours)
    contours_list.append([final_set_of_subcontours])
    # print('final_set_of_subcontours', len(final_set_of_subcontours))
    images_predContours_dict[file_] = final_set_of_subcontours
    areas_list.append([areas])
    peris_list.append([peris])
    final_subcontours+=final_set_of_subcontours
    final_areas+=areas
    final_peris+=peris
    #cv2.drawContours(img, final_set_of_subcontours, -1, 255, 1)

  #cv2.drawContours(img, final_set_of_subcontours, -1, (255,0,0), 1)
  #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  print('done contour')
  #cv2.namedWindow('Final',cv2.WINDOW_AUTOSIZE)
  #cv2.imshow('Final', img)
  #cv2.resizeWindow('Final', 1000,1000)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()	
    
  return contours_list, areas_list, peris_list,final_subcontours, final_areas, final_peris, image_stack,images_predContours_dict,concave_points_dict,contour_splitlines_dict,contour_splitlines_img_dict,angles__contours_image_dict,concaves__contours_image_dict,img_cv2_hull_dict,img_cv2_contours_dict,concavePT_angle_image_dict

#print(image_files_list)
#contours_list, areas_list, peris_list,final_subcontours, final_areas, final_peris, image_stack,images_predContours_dict,concave_points_dict,contour_splitlines_dict,contour_splitlines_img_dict,angles__contours_image_dict,concaves__contours_image_dict,img_cv2_hull_dict,img_cv2_contours_dict,concavePT_angle_image_dict = contour_raw(image_files_list)
# contours_list, areas_list, peris_list,final_subcontours, final_areas, final_peris, image_stack,images_predContours_dict,concave_points_dict,contour_splitlines_dict,contour_splitlines_img_dict,angles__contours_image_dict,concaves__contours_image_dict,img_cv2_hull_dict,img_cv2_contours_dict,concavePT_angle_image_dict = contour_raw([image_files_list[200]])

"""## **Target**"""

### FEATURES ###

def num_of_pixels_in_contour(final_contours,img):
#https://stackoverflow.com/questions/33234363/access-pixel-values-within-a-contour-boundary-using-opencv-in-python
	
	num_pixels = []
	for i in range(len(final_contours)):
		# Create a mask image that contains the contour filled in
		cimg = np.zeros_like(img)
		cv2.drawContours(cimg, final_contours, i, color=255, thickness=-1)

		# Access the image pixels and create a 1D numpy array then add to list
		pts = np.where(cimg == 255)
		size = len(pts)
		num_pixels.append(size)
		
	return num_pixels

def get_rad_cen(final_contours):
	
	radii = []
	centers = []
	for cnt in final_contours:
		(x,y), radius = cv2.minEnclosingCircle(cnt)
		radii.append(radius)
		centers.append((x,y))
		
	return radii, centers	
	
def isconvex(final_contours):

	convexity = []
	for cnt in final_contours:
		k = cv2.isContourConvex(cnt)
		convexity.append(k)
	return convexity
	
def get_aspectratio(final_contours):

  aspect_ratio = []
  for cnt in final_contours:
    cnt = np.array(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    asprat = 0.1*100*(w)/h	
    aspect_ratio.append(asprat)
    
  return aspect_ratio
	
def get_solidity(final_contours):
	
	solidity = []
	for cnt in final_contours:
		area = cv2.contourArea(cnt)
		hull = cv2.convexHull(cnt)
		hull_area = cv2.contourArea(hull)
		soli = 0.1*100*(area)/hull_area
		solidity.append(soli)
		
	return solidity

def get_extent(final_contours):

  extent = []
  for cnt in final_contours:
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    ext = 0.1*100*(area)/rect_area
    extent.append(ext)
    
  return extent
	
def get_centroid(final_contours, centers):
	
	i = 0
	centroid = []
	while i < len(final_contours):
		cnt = final_contours[i]
		cX, cY = centers[i]
		M = cv2.moments(cnt)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		
		centroid.append([cX, cY])	
		i = i+1
		
	
	return centroid
	
def get_centroid_dis_fun(final_contours, centroid):
	
	fun = []
	for i in range(len(final_contours)):
		cnt = final_contours[i]
		hull = cv2.convexHull(cnt)
		cen = centroid[i]
		
		d = 0
		a =0
		b = 0
		#print(cnt)
		for j in range(len(hull)):
			x = hull[j][0][0]
			y = hull[j][0][1]
			
			a = (x-cen[0])**2
			b = (y-cen[1])**2	
			
			t = (a+b)**0.5
			d = d+t
		
		fun.append(int(d))
	
	print(fun[100:120])
	return fun
	
def get_circularity_ratio(areas, peris):
	
	cr = []
	for i in range(len(areas)):
		a = areas[i]
		p = peris[i]
		
		r = a*100.0/(p*p)
		cr.append(r)
		
		
	return cr
	
def get_rectangularity(final_contours):
	
	rect = []
	for i in range(len(final_contours)):
		cnt = np.array(final_contours[i])
		x,y,w,h = cv2.boundingRect(cnt)
		a_r = w*h
		a = cv2.contourArea(cnt)
		rat = a/a_r
		
		rect.append(rat)
		
	return rect
	
def get_convexity(final_contours):
	
	con = []
	for i in range(len(final_contours)):
		cnt = final_contours[i]
		hull = cv2.convexHull(cnt)
		p = cv2.arcLength(cnt, True)
		ph = cv2.arcLength(hull, True)
		
		rat = ph/p
		con.append(rat)
		
	return con
	
		
def get_features(final_contours):
	
## Advanced Properties https://gurus.pyimagesearch.com/lesson-sample-advanced-contour-properties/
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html			
	radii, centers = [[],[]]#get_rad_cen(final_contours) # Minimum Enclosing Circle #
	convexity = []# isconvex(final_contours) # Is Convex #
	aspect_ratio = get_aspectratio(final_contours) # Aspect Ratio# *aspect ratio = image width / image height*
	solidity = []# get_solidity(final_contours) # Solidity # *solidity = contour area / convex hull area*
	extent = get_extent(final_contours) # Extent #  *extent = shape area / bounding box area*
	#num_pixels = num_of_pixel s_in_contour(final_contours,image)
	num_pixels = []
	#moments = get_moments(final_contour)
	print('done getting features')
	
	return radii, centers, convexity, aspect_ratio, extent, solidity, num_pixels
	
def feature_advanced(final_contours, radii, centers, convexity, aspect_ratio, extent, solidity, num_pixels, areas, peris):
	
	#https://www.slideshare.net/noorwzr/1-shape-features
	conv = []# get_convexity(final_contours)
	circu = get_circularity_ratio(areas, peris)
	rectan = get_rectangularity(final_contours)
	centroid = []#get_centroid(final_contours, centers)
	M = []#get_centroid_dis_fun(final_contours, centers)
	m = []#np.asarray(M)
	
	LC_set1	= areas + peris
	LC_set2 = circu + rectan + aspect_ratio + extent
	#print(type(LC_set2))
		
	print('done advanced')
		
#	return LC_set1,LC_set2
	return 	circu, rectan, M, conv, LC_set1, LC_set2

def __loss(orig, pred, los ='mse'):
  if los == 'mse':
    mse = np.mean((orig - pred) ** 2) 
    return mse

  elif los == 'psnr':
    mse = np.mean((orig - pred) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

  else:
    l = orig-pred

def find_accuracy():
	## Calculate Accuracy
	loss = 0
	count = 0
	for imgkey in image_files_list:
		file_ = imgkey.split('/')[-1]
		try:
			cont_pred = images_predContours_dict[file_]
			cont_actual = image_contour_dict[file_]

			img1 = cv2.imread(imgkey)
			cv2.drawContours(img1,cont_pred,-1,0,-1)
			_, thresh1 =  cv2.threshold(img1,20,255,cv2.THRESH_BINARY)
			thresh1 = np.array(thresh1)
			img2 = cv2.imread(imgkey)
			cv2.drawContours(img2,cont_actual,-1,0,-1)
			_, thresh2 =  cv2.threshold(img2,20,255,cv2.THRESH_BINARY)
			thresh2 = np.array(thresh2)

			loss += __loss(thresh1, thresh2)

			count+=1
		except:
			continue

	print(loss)
	print(loss/count)

"""## **VISUALIZATION OF IMAGE WITH CONTOURS**"""

def visualize_contours(imgkey,images_predContours_dict, savepath):
	
	"""
	concaves = concave_points_dict[image_files_list[index]]
	concaves_list = []
	# print(concaves)

	for i in range(len(concaves)):
		con = concaves[i]
		# print(con)
		for j in range(len(con)):
			if len(con[j]) ==0:
				continue
			else:
				ele = con[j]
				concaves_list.append(np.array([ele]))
		  

	# print(len(concaves_list))

	con_ang_dict = concavePT_angle_image_dict[image_files_list[index]]
	req_list = []
	for ele in con_ang_dict.items():
		key = ele[0]
		val = ele[1]

		if int(val)==45:
			req_list.append(key)

	req_list_ =[]
	for j in range(len(concaves_list)):
		ele = concaves_list[j]
		ele = list(ele)
		req_list_.append(np.array([ele]))

	print('Number of concave points in image', len(req_list_))
	"""
	image = cv2.imread(imgkey)
	file_ = imgkey.split('/')[-1]
	#cv2.imshow('actual image', image)
	cont_pred = images_predContours_dict[file_]
	print('Number of Predicted Contours in Image', len(cont_pred))
	cv2.drawContours(image,cont_pred,-1,(0,0,255),1)
	#cv2.imshow('contours', image)
	cv2.imwrite(savepath, image)

def draw_actual_contours(imgkey,image_contour_dict, file_, savepath):
	image = cv2.imread(imgkey)
	cont_real = image_contour_dict[file_]
	print('Number of Actual Contours in Image', len(cont_real))
	cv2.drawContours(image,cont_real,-1,(0,0,255),1)
	#cv2.imshow('contours', image)
	cv2.imwrite(savepath, image)


#for i in range(len(image_files_list)):
#	savepath = path + '/' + 'result' + str(i) +'.png'
#	visualize_contours(image_files_list,images_predContours_dict, i, savepath)

def get_contours_pred(imgkey, images_predContours_dict):
	file_ = imgkey.split('/')[-1]	
	cont_pred = images_predContours_dict[file_]
	
	return cont_pred

def get_contours_real(imgkey, image_contour_dict, file_):
	image = cv2.imread(imgkey)
	cont_real = image_contour_dict[file_]
	
	return cont_real

def cont_to_dict(contours):
	dictionary = {}
	

"""## **Visualize Each Contour**"""
def visualize_each_contour(contour):
	list_of_points = []
	for i in range(len(contour)):
	  ele = contour[i]
	  print(ele)
	  ele = tuple(ele[0])
	  list_of_points.append(ele)

	#print(list_of_points)

	points = list_of_points
	fig, ax = plt.subplots()	
	ax.scatter(*zip(*points))
	plt.show()
	


## Utils for analysis
def KMeans_algo(df, nclus):
	kmeans = KMeans(n_clusters= nclus, random_state=0).fit(df)
	print('done Kmeans')
	labels_km = kmeans.labels_
	
	return labels_km

def PCA_anal(df):
	
	col = df.columns
	#df = StandardScaler().fit_transform(df)
	data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
	#print(df[1:10])
	pca = PCA(n_components=6)
	principalComponents = pca.fit_transform(data_scaled)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['pca 1', 'pca 2', 'pca 3', 'pca 4', 'pca 5', 'pca 6'])
	principalDF = pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6'])
	print(principalDF.head())
	
	return principalDf

def pairPlots(df):
#https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
	# Pair-wise Scatter Plots
	cols =['Perimeter', 'Circularity_ratio', 'Rectangularity_ratio','Aspect_ratio','Extent','Area','Radii', 'Convexity']
	pp = sns.pairplot(df[cols], size=1.8, aspect=1.5,
			          plot_kws=dict(edgecolor="k", linewidth=0.5),
			          diag_kind="kde", diag_kws=dict(shade=True))

	pp.savefig("pair_plot.png")
	fig = pp.fig 
	fig.subplots_adjust(top=0.93, wspace=0.3)
	t = fig.suptitle('Pairwise Plots', fontsize=10)
 
def plot_features_modelop(feature_dict_list,contour_list, labels_list):
	
  feature1_l1 = []
  feature1_l2 = []
  feature1_l3 = []
  feature2_l1 = []
  feature2_l2 = []
  feature2_l3 = []

  for i in range(len(contour_list)):
    areas = feature_dict_list[i]['areas']
    peris = feature_dict_list[i]['peris']
    aspect_ratio = feature_dict_list[i]['aspect_ratio']
    circu = feature_dict_list[i]['circularity_ratio']
    rectan = feature_dict_list[i]['rectangularity_ratio']
    extent = feature_dict_list[i]['extent']
    final_contours = contours_list[i][0]
    labels_km = labels_list[i][0]

    fea1 = np.sqrt(np.array(areas))
    fea2 = np.divide(np.array(peris),fea1)

    LC_set1 = np.array(areas)#	+np.array(peris)+np.array(aspect_ratio)
    LC_set2 = np.array(peris) #np.array(circu)+np.array(rectan)+np.array(extent) 

    contour_label_dict = {}

    for i in range(len(final_contours)):
      contour_label_dict[tuple(final_contours[i].ravel())] = labels_km[i]
      
    l1 = 0
    l2 = 0
    l3 = 0
    t = 0

    labels1 = []
    labels2 = []
    labels3 = []
    
    for cnt in final_contours:
      lab = contour_label_dict[tuple(cnt.ravel())]
      if lab == 1:
        #cv2.drawContours(img, [cnt], -1, (0,255,0), 1)	#Green
        feature1_l1.append(LC_set1[t])
        feature2_l1.append(LC_set2[t])
        labels1.append(lab)
        l1 = l1+1
        
      elif lab == 2:
        #cv2.drawContours(img, [cnt], -1, (0,0,255), 1)	#Black
        feature1_l2.append(LC_set1[t])
        feature2_l2.append(LC_set2[t])
        labels2.append(lab)
        l2 = l2+1
        
      else:
        #cv2.drawContours(img, [cnt], -1, (255,0,0), 1)	#Blue
        feature1_l3.append(LC_set1[t])
        feature2_l3.append(LC_set2[t])
        labels3.append(lab)
        l3 = l3+1
        
      t = t+1
    
  print('done labelling')
    
  return feature1_l1,feature1_l2, feature1_l3,feature2_l1, feature2_l2, feature2_l3
	

def cluster_analysis(feature1_l1,feature1_l2, feature1_l3,feature2_l1, feature2_l2, feature2_l3):

	plt.scatter(feature1_l1, feature2_l1, s=1, c = 'b',marker = 'o')
	plt.scatter(feature1_l2, feature2_l2, s=1, c = 'g',marker = 'o')
	plt.scatter(feature1_l3, feature2_l3, s=1, c = 'k',marker = 'o')	
	plt.xlabel('feature1')
	plt.ylabel('feature2')
	plt.show()
	print('done analysis')

def cluster_contours(contours_list, areas_list, peris_list, image_stack):
  labels_list = []
  cluster_img_stack = []
  feature_dict_list = []

  for i in range(len(contours_list)):
    final_contours = contours_list[i][0]
    areas = areas_list[i][0]
    peris = peris_list[i][0]
    img = image_stack[i]

    radii, centers, convexity, aspect_ratio, extent, solidity, num_pixels = get_features(final_contours)
    circu, rectan, M, conv, data1, data2 = feature_advanced(final_contours,radii, centers, convexity, aspect_ratio, extent, solidity, num_pixels, areas, peris)
    feature_dict ={'areas':areas, 'peris':peris, 'radii':radii, 'centers':centers, 'aspect_ratio':aspect_ratio, 'extent':extent, 'solidity':solidity, 'circularity_ratio':circu, 'rectangularity_ratio':rectan}
    feature_dict_list.append(feature_dict)
    df = pd.DataFrame(list(zip(peris,circu,rectan,aspect_ratio, extent, areas)), columns =['Perimeter', 'Circularity_ratio', 'Rectangularity_ratio','Aspect_ratio','Extent','Area'])

    #principalDf = PCA_anal(df)
    #LC_set1 = areas+peris+aspect_ratio #principalDf['pca 1']
    #LC_set2 = circu+rectan+extent #principalDf['pca 2']

    nclus = 3 # number of clusters
    labels_km = KMeans_algo(df, nclus)
    labels_list.append([labels_km])

    cluster_img_stack.append(img)
    
    
  return contours_list, labels_list, feature_dict_list, cluster_img_stack

def clustering_and_analysis(contours_list,areas_list, peris_list, image_stack):
	contours_list, labels_list, feature_dict_list, cluster_img_stack = cluster_contours(contours_list,areas_list, peris_list, image_stack)
	feature1_l1,feature1_l2, feature1_l3,feature2_l1, feature2_l2, feature2_l3 = plot_features_modelop(feature_dict_list,contours_list, labels_list)
	print('drawing Pred Plot')
	cluster_analysis(feature1_l1,feature1_l2, feature1_l3,feature2_l1, feature2_l2, feature2_l3)
	

#clustering_and_analysis(contours_list,areas_list, peris_list, image_stack)

#array([[[116,   1]], [[118,   1]], [[121,   3]], [[121,   0]]]   ->>> Contour format
## Target Inputs
def dict_to_array(list_of_dicts):
  # {'x': 72, 'y': 11} to [72,11]
	list_of_arrs = []
	for dictionary in list_of_dicts:
		x = dictionary['x']
		y = dictionary['y']
		list_of_arrs.append([[x, y]])  #IMPORTANT -> This is the form required by cv2.drawContour()

	return list_of_arrs

def load_contour_from_json(jsonPath):
	f = open(jsonPath)
	img_annots_list = json.load(f)

	target_contour_list = []
	target_labels_list = []
	areas_list = []
	peris_list = []
	image_contour_dict = {}

	for i in range(len(img_annots_list)):
		d = img_annots_list[i]
		file_ = d["External ID"]
		file_ = file_.split('/')[-1]
		count = 0

		contours_image = []
		try:
			cell1 = img_annots_list[i]["Label"]["cell1"]
			
			for p in range(len(cell1)):
				cont = cell1[p]["geometry"]
				contours_image.append(np.array(dict_to_array(cont)))
				target_contour_list.append(dict_to_array(cont))
				target_labels_list.append(0)
				area = cv2.contourArea(np.array(dict_to_array(cont)))
				peri = cv2.arcLength(np.array(dict_to_array(cont)), True)
				areas_list.append(area)
				peris_list.append(peri)

		except:
			#pass
			count = count+1
		try:
			cell2 = img_annots_list[i]["Label"]["cell2"]
			for q in range(len(cell2)):
				cont = cell2[q]["geometry"]
				contours_image.append(np.array(dict_to_array(cont)))
				target_contour_list.append(dict_to_array(cont))
				target_labels_list.append(1)
				area = cv2.contourArea(np.array(dict_to_array(cont)))
				peri = cv2.arcLength(np.array(dict_to_array(cont)), True)
				areas_list.append(area)
				peris_list.append(peri)

		except:
			#pass
			count = count+1

		try:
			cell3 = img_annots_list[i]["Label"]["cell3"]
			for r in range(len(cell3)):
				cont = cell3[r]["geometry"]
				contours_image.append(np.array(dict_to_array(cont)))
				target_contour_list.append(dict_to_array(cont))
				target_labels_list.append(2)
				area = cv2.contourArea(np.array(dict_to_array(cont)))
				peri = cv2.arcLength(np.array(dict_to_array(cont)), True)
				areas_list.append(area)
				peris_list.append(peri)

		except:
			#pass
			count = count+1

		image_contour_dict[file_] = contours_image
		print(len(target_contour_list))
		print(len(areas_list))
		print(len(peris_list))
		#target_feature_dict=get_feature_dict_list(target_contour_list, areas_list, peris_list)

		#return target_contour_list, target_labels_list, areas_list, peris_list, target_feature_dict, image_contour_dict
	return image_contour_dict

def get_centers_list(contours):
	rad, centers = get_rad_cen(contours)	
	return centers

def one_one_correspondence(array1, array2):
	array1 = np.array(get_centers_list(array1))
	array2 = np.array(get_centers_list(array2))
	print('ar1', array1)
	print('ar2', array2)

	false_negative = 0
	true_positive = 0
	true_negative = 0
	false_positve = 0
	i = 0
	while(i < len(array2)):
		b = True
		j = 0
		while(j < len(array1) and b):
			ele1 = array1[j]
			ele2 = array2[i]
			dis = euclidean(ele1[0],ele1[1], ele2[0], ele2[1])
			if dis<10:
				true_positive+=1
				array1 = np.delete(array1, j, axis=0)
				array2 = np.delete(array2, i, axis=0)
				b = False

			else:
				j+=1


		if b:
			true_negative+=1
			i+=1


	false_positve = len(array1) 

	return true_positive, false_positve, true_negative, false_negative
	

def f1_score_finder(array1, array2):
    true_positive, false_positive, true_negative, false_negative = one_one_correspondence(array1, array2)
    print('true_positive', true_positive)
    print('true_negative', true_negative)
    print('false_positve', false_positive)
    print('false_negative', false_negative)

    precision = true_positive/(true_positive+false_positive)
    if (true_positive+false_negative)==0:
        print('no recall')
        print('No F1')
        F1Score = 'Not defined'
        recall = 'Not defined'
    else:
        recall = true_positive/(true_positive+false_negative)
        F1Score = 2*precision*recall/(precision+recall)

    return precision, recall, F1Score


def main(args):
	
	print("RUNNING !!")
	input_path = args.inputpath
	output_path = args.outputpath
	targetPresent = args.targetPresent
	json_path = args.jsonPath	#'/media/aiswarya/New Volume/My_works/CCBR-IITM-Thesis/data.json'
	image_files_list = get_image_inputs(input_path)
	
	contours_list, areas_list, peris_list,final_subcontours, final_areas, final_peris, image_stack,images_predContours_dict,concave_points_dict,contour_splitlines_dict,contour_splitlines_img_dict,angles__contours_image_dict,concaves__contours_image_dict,img_cv2_hull_dict,img_cv2_contours_dict,concavePT_angle_image_dict = contour_raw(image_files_list)
	dictionary = {}
	for i in range(len(image_files_list)):
		imgkey = image_files_list[i]
		print(imgkey)
		#fil = img_cv2_contours_dict[imgkey]
		file_ = imgkey.split('/')[-1]
		dict_key = file_
		num = file_.split('.')[0][-1]
		savepath = output_path + '/' + 'result' + str(num) +'.png'
		visualize_contours(imgkey,images_predContours_dict, savepath)
		#dict_val = cont_to_dict(cont_pred)

		#dictionary[dict_key] = dict_val
	
	if targetPresent:
		image_contour_dict = load_contour_from_json(json_path)
		cont_pred = get_contours_pred(imgkey, images_predContours_dict)
		file__ = list(image_contour_dict.keys())[0]
		cont_real = get_contours_real(imgkey, image_contour_dict, file__)
		savepath_ = output_path + '/' + 'real' + str(num) +'.png'
		draw_actual_contours(imgkey, image_contour_dict, file__, savepath_)

		precision, recall, F1_score = f1_score_finder(cont_pred, cont_real)
		print('precision', precision)
		print('recall', recall)
		print('F1', F1_score)
	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Nissl Cell Segmentation')
	parser.add_argument('--inputpath', default = '\home', type = str, help = 'path to the input files')
	parser.add_argument('--outputpath', default = '\home', type = str, help = 'path to the output files')
	parser.add_argument('--targetPresent', default = False, type = bool, help = 'If the target cell segmentation is available, then this variable is set to true')
	parser.add_argument('--jsonPath', default = '\home', type = str, help = 'Given contours in json format, apply on image')

	args = parser.parse_args()
	main(args)
	
	
	

