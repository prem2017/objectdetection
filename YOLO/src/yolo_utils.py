# -*- coding: utf-8 -*- 

""" ©Prem Prakash
	General YOLO utility module 

	Credits: one section of code is taken/adapted from 
				Source: [Taken From Github] (https://github.com/HeroKillerEver/coursera-deep-learning)
"""

import pdb


import colorsys
import imghdr
import os
import random
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

import util 
from util import logger



#******************** Constants
K_MODEL_IMAGE_SIZE = (256, 256) # (Width, Height) # for now it is same as true but should check (256, 256)
get_model_img_size = lambda: K_MODEL_IMAGE_SIZE

K_CLASSES = 2
get_num_classes = lambda: K_CLASSES
get_class_names = lambda: ('Beet', 'Thistle')
get_class_colors = lambda: ('Yellow', 'Red')

# Grid and anchors information
K_GRID_SHAPE = (4, 4)
get_grid_shape = lambda: K_GRID_SHAPE 
K_ANCHORS = 1
get_num_anchors = lambda: K_ANCHORS



#******************** Taken from https://github.com/HeroKillerEver/coursera-deep-learning
def read_classes(classes_path):
	with open(classes_path) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names


def read_anchors(anchors_path):
	with open(anchors_path) as f:
		anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		anchors = np.array(anchors).reshape(-1, 2)
	return anchors


def generate_colors(class_names):
	hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.
	return colors

def scale_boxes(boxes, image_shape):
	""" Scales the predicted boxes in order to be drawable on the image"""
	width, height = image_shape
	image_dims = np.array([[width, height, width, height]]) # K.stack([height, width, height, width])
	boxes = boxes * image_dims
	return boxes


def load_and_resize_image(img_path, model_image_size):
	"""Reads the image and resizes it.
		
		Parameters:
		-----------
			img_path (str): fullpath to the image where it is located.
			model_image_size (tuple): the dimension (width, height) of image which goes to model. 
									  Note: here that pil-images have first dimension width and second height 

		Returns:
		--------
			true_img_size (tuple): the true (width, height) of original image.
			image_data (numpy.ndarray): the resized image in shape (H x W x C)
	"""
	image_type = imghdr.what(img_path)
	image = Image.open(img_path)
	resized_image = image.resize(model_image_size, Image.BICUBIC) # NOTE:  (width, height).
	image_data = np.array(resized_image, dtype='float32') #  this converts to (height x width x channel)
	image_data /= 255.
	# image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
	true_img_size = image.size
	return true_img_size, image_data





def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
	"""Draw boxes with the class-name, confidence, and box-boundary 

		Parameters:
		-----------
			image (PIL.Image): the pil-image (of shape (W x H x C)) on which boxes will be drawn.
			out_scores (list): confidence (conf * class_prob) corresponding to each of the boxes detected.
			out_boxes (list[list]): list of boxes with (x1, y1, x2, y2) coordinates.
			out_classes (list): class corresponding to each box.
			class_names (list): all the classes that the model can detect.
			colors (list): list of colors corresponsing to each unique class. 

		Returns:
			None: 
	"""
	
	font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
	thickness = (image.size[0] + image.size[1]) // 300

	for i, c in reversed(list(enumerate(out_classes))):
		predicted_class = class_names[c]
		box = out_boxes[i]
		score = out_scores[i]

		label = '{} {:.2f}'.format(predicted_class, score)

		draw = ImageDraw.Draw(image)
		label_size = draw.textsize(label, font)

		# TODO: replce by  `left, top, right, bottom = box`
		# top, left, bottom, right = box
		left, top, right, bottom = box

		area = (right - left) * (bottom - top)
		print('area = ', abs(area))
		if abs(area) < 1e-2:
			continue

		left = max(0, np.floor(left + 0.5).astype('int32'))
		top = max(0, np.floor(top + 0.5).astype('int32'))

		# 
		right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
		bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

		print(label, (left, top), (right, bottom))

		if top - label_size[1] >= 0:
			text_origin = np.array([left, top - label_size[1]])
		else:
			text_origin = np.array([left, top + 1])

		# A good redistributable image drawing library.
		for i in range(thickness):
			draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c]) # Rectangular box around the object
		draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='white') #  colors[c] Box around the text
		draw.text(text_origin, label, fill=(0, 0, 0), font=font) # The text
		del draw


def draw_boxes_raw(image, boxes, class_names = ['Beet', 'Thistle'], colors=['Yellow', 'Red']):
	""" Temporary for checking that the created dataset works to draw boxes around the objects.
		
		Parameters:
		-----------
			image (PIL.Image): A loaded image is given whose shape should be in (WxHxC)
			boxes (list[list]: rectangular boxes with coordinates ((x1, y1), (x2, y2)) and respective class
		
		Returns:
		--------
			None: Draws the boxes and saves them 
	"""
	print('[Called]')
	font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
	print('[Font] = ', font)
	thickness = (image.size[0] + image.size[1]) // 300


	boxes = np.array(boxes)
	out_boxes = boxes[:, :4]
	out_classes = boxes[:, 4]
	out_scores = [1 for i in out_classes] # For Scores



	for i, c in reversed(list(enumerate(out_classes))):
		predicted_class = class_names[c]
		box = out_boxes[i]
		score = out_scores[i]

		label = '{} {:.2f}'.format(predicted_class, score)

		draw = ImageDraw.Draw(image)
		label_size = draw.textsize(label, font)

		left, top, right, bottom = box


		
		left = max(0, np.floor(left + 0.5).astype('int32'))
		top = max(0, np.floor(top + 0.5).astype('int32'))

		# 
		right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
		bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
		
		print(label, (left, top), (right, bottom))

		if top - label_size[1] >= 0:
			text_origin = np.array([left, top - label_size[1]])
		else:
			text_origin = np.array([left, top + 1])

		# My kingdom for a good redistributable image drawing library.
		for i in range(thickness):
			draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])

		draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
		draw.text(text_origin, label, fill=(0, 0, 0), font=font)
		del draw
	return 


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
	"""Filters YOLO boxes by thresholding on object and class confidence.
	
		Arguments:
			box_confidence (numpy.ndarray): shape (GR x GC x A x 1)
			boxes (numpy.ndarray): shape (GR x GC x A x 4) Note: that the box arrya can represnt 
					 (x1, y1, x2, y2) or (obj_cx, obj_cy, obj_w, obj_h) or (delta_cx, delta_cy, delta_w, delta_h) 
					 does not affect the result since the method only filter the results based on confidence  
			box_class_probs (numpy.ndarray): shape (GR x GC x A x 2)
			threshold (float): if [highest class probability score < threshold], then get rid of the corresponding box
			
		Returns:
			scores (list): shape (None,), containing the class probability score for selected boxes
			boxes (list):  shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
			classes (list): shape (None,), containing the index of the class detected by the selected boxes
			
		Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
		For example, the actual output size of scores would be (10,) if there are 10 boxes.
	"""
	
	# TOOD: may be perform only on class probability rahter than (box_confidence * box_class_probs)
	# Element-wise multiplication to compute the confidence i.e. obj/non-obj * class_probs
	box_scores = np.multiply(box_confidence, box_class_probs)
	
	# Find the box_classes thanks to the max box_scores, keep track of the corresponding score
	box_classes = np.argmax(box_scores, axis=-1)
	box_class_scores = np.max(box_scores, axis=-1)

	# Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	# same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
	filtering_mask = np.greater_equal(box_class_scores, threshold)
	
	# Apply the mask to scores, boxes and classes
	scores = box_class_scores[filtering_mask] # shape = (?,) i.e. is list of length = filtering_mask.sum() meaning for all element for which mast is positive 
	boxes = boxes[filtering_mask] # shape = (?, 4) ? is represent boxes whose mask value is positive that are the filtered ones
	classes = box_classes[filtering_mask] # shape = (?,) 
	
	return scores, boxes, classes


def iou(box1, box2):
	"""Implement the intersection over union (IoU) between box1 and box2
	
		Arguments:
			box1 -- first box, list object with coordinates (x1, y1, x2, y2)
			box2 -- second box, list object with coordinates (x1, y1, x2, y2)
		Returns:
			Jaccard's distance/IOU value
	"""

	# Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
	# pdb.set_trace()
	xi1 = max(box1[0], box2[0])
	yi1 = max(box1[1], box2[1])
	xi2 = min(box1[2], box2[2])
	yi2 = min(box1[3], box2[3])
	inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
	
	if inter_area <= 0:
		msg = '\nbox1 = {}, box2 = {}\ninter_area = {}'.format(box1, box2, inter_area)
		logger.info(msg)
		return 0
	
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
	union_area = box1_area + box2_area - inter_area

	iou = inter_area / union_area

	return iou





#******************** Additions
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 9, iou_threshold = 0.5):
	"""Applies Non-max suppression (NMS) to set of boxes
	
		Parameters:
		-----------
			scores (list): output of yolo_filter_boxes()
			boxes (list): output of yolo_filter_boxes() that have been scaled to the image size (see later) and are [x1, y1, x2, y2]
			classes (list): output of yolo_filter_boxes()
			max_boxes (int): maximum number of predicted boxes we want to keep
			iou_threshold (float): "intersection over union" threshold used for NMS filtering
			
		Returns:
		--------
			scores (list): of shape (, None), predicted score for each box
			boxes (list): of shape (4, None), predicted box coordinates
			classes (list): of shape (, None), predicted class for each box
			
		Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
		function will transpose the shapes of scores, boxes, classes. This is made for convenience.
	"""
	fscores = []
	fboxes = []
	fclasses = []
	print(scores.shape, boxes.shape, classes.shape)

	def append_in_final(i):
		fscores.append(scores[i])
		fboxes.append(boxes[i])
		fclasses.append(classes[i])

	unique_classes = np.unique(classes)
	print('unique_classes = ', unique_classes)
	print('\nclasses = ', classes)
	for label in unique_classes:
		# Get all the indices which has same label and check if 
		indices = list(np.where(classes == label)[0])
		# len(indices) <= 1 
		if len(indices) == 1:
			append_in_final(indices[0])
			continue	

		i = 0
		remaining_indices = deepcopy(indices)
		while i < len(indices):
			print('\n[First-While] i = {}, indices = {}, remaining_indices = {}'.format(i, indices, remaining_indices))
			j = i + 1
			if indices[i] not in remaining_indices:
				i += 1
				continue

			while j < len(indices):
				if indices[j] not in remaining_indices:
					j += 1
					continue

				iou_val = iou(boxes[indices[i]], boxes[indices[j]])
				print('indices = {}, remaining_indices = {}, indices[i] = {}, indices[j] = {}, iou_val = {}'.format(indices, remaining_indices, (i, indices[i]), (j, indices[j]), iou_val))
				# print(f'i = {i}, boxes[indices[i]] = {boxes[indices[i]]}, \n j = {j}, boxes[indices[j]], {boxes[indices[j]]}')
				# print(f'scores[indices[i]] = {scores[indices[i]]}  scores[indices[j]] = {scores[indices[j]]}')
				if iou_val >= iou_threshold:
					# pdb.set_trace()
					if scores[indices[i]] < scores[indices[j]]:
						remaining_indices.remove(indices[i])
					else:
						remaining_indices.remove(indices[j])
				j += 1 # End of inner while loop 

			i += 1	# End of outer while loop

		print('\n[Final]i = {}, indices = {}, remaining_indices = {}'.format(i, indices, remaining_indices))				
		for index in remaining_indices:
			append_in_final(index)

	# pdb.set_trace()

	scores = np.array(fscores)
	boxes = np.array(fboxes)
	classes = np.array(fclasses)
	return scores, boxes, classes


#******************** Next four method to create target from box and reverse it
def delta_obj_center(obj_center, cell_center, cell_size):
	"""Computes the delta (difference) value of object center from cell center using tanh (range [-1, 1]) for shift of object 
	   center from cell center where max shift of cell_width/2 equals -1 for the left and 1 for the right along x
	   and similarly for y

		Parameters:
		-----------
			obj_center (tuple): center of the object i.e. x, y
			cell_center (tuple): center fo the cell i.e. x, y
			cell_size (tuple): widht and height of cell to compute the shift of object center from cell center

		Returns:
		--------
			tuple: the computed shift with respect to tanh  
		
	"""
	# print(obj_center, cell_center, cell_size)
	obj_cx, obj_cy = obj_center
	cell_cx, cell_cy = cell_center
	cell_width, cell_height = cell_size 

	delta_cx = np.arctanh((obj_cx - cell_cx) * (2 / cell_width) + 1e-6) # 1e-6 to avoid nan 
	delta_cy = np.arctanh((obj_cy - cell_cy) * (2 / cell_height) + 1e-6)
	return delta_cx, delta_cy

	
def inverse_delta_obj_center(delta_coord, cell_center, cell_size):
	"""To restore object center from predicted delta_cx, delta_y

		Parameters:
		-----------
			delta_coord (tuple): the shift of the object center 
			cell_center (tuple): center fo the cell i.e. x, y
			cell_size (tuple): widht and height of cell to compute the shift of object center from cell center
		
		Returns:
		--------
			tuple: the (x, y) center of the object

	
	"""
	delta_cx, delta_cy = delta_coord
	cell_cx, cell_cy = cell_center 
	cell_width, cell_height = cell_size

	obj_cx = cell_cx + (cell_width / 2) * np.tanh(delta_cx)
	obj_cy = cell_cy + (cell_height / 2) * np.tanh(delta_cy)
	return obj_cx, obj_cy


def delta_obj_size(object_image_size, model_image_size):
	"""To compute the delta (scale) value of object (width, height) fromthe image (width, height) using sigmoid (range [0, 1]).
		Since sigmoid limits values between 0, 1 and it is less of equal to image width or height thus the limit of 0-1 is one of
		the suitable option.

		Parameters:
		-----------
			  object_image_size (tuple): width and height of the object. 
			  model_image_size (tuple): width and height of model (prediction is on rescaled image)

		Returns:
		--------
			tuple: the scaled width and height w.r.t. model
		
		
	"""
	obj_w, obj_h = object_image_size
	model_w, model_h = model_image_size

	delta_w = -(np.log((model_w / obj_w) + 1e-6 - 1)) # 1e-6 to avoid nan 
	delta_h = -(np.log((model_h / obj_h) + 1e-6 - 1))

	return delta_w, delta_h

def inverse_delta_obj_size(delta_image_size, model_image_size):
	"""Restores the object width and height from predicted delta values

		Parameters:
		-----------
			delta_image_size (tuple): delta_w, delta_h predicted from the wodel
			model_image_size (tuple): width and height which is the input dimension to the model 
		
		Returns:
		--------
			tuple: the width and height of the object.

	"""
	delta_w, delta_h = delta_image_size
	model_w, model_h = model_image_size

	obj_w = model_w * (1 / (1 + np.exp(-delta_w))) 
	obj_h = model_h * (1 / (1 + np.exp(-delta_h)))

	return obj_w, obj_h



def box_coordinates(object_center, object_size, model_image_size):
	"""Given object center and size get the coordinates of two extremes from top-left to bottom-right.

		Parameters:
		-----------
			object_center (tuple): the center (x, y) of the object
			object_size (tuple): width and height of the object
			model_image_size (tuple): width and height which is the input dimension to the model 

		Returns:
		--------
			tuple: two extremes of the object (x1, y1, x2, y2)
	"""
	obj_cx, obj_cy = object_center
	obj_w, obj_h = object_size

	width, height = model_image_size

	x1, y1 = max(0, obj_cx - (obj_w / 2)), max(0, obj_cy - ( obj_h / 2))
	x2, y2 = min(width, obj_cx + (obj_w / 2)), min(height, obj_cy + ( obj_h / 2)) 

	return x1, y1, x2, y2


def construct_target(coord_labels, true_image_size, model_image_size, num_cls, grid_shape, anchors=1):
	"""Construct the target of shape: grid_shape (GR, GC) x (anchors (K) * (1 + Coordinate (x, y, w, h): 4 + #Classes (C)))
	   Shape: GR x GC x (A*5+C) i.e for G =4, A = 1, C = 2 the shape is 4x4x7.  
		

		Parameters:
		-----------
			coord_labels (list[list]): contains the list of localised coordinates of all the present object with first 
						 four elements are min (x1, y1), max (x2, y2) cordinates, then label, etc.- is not of concern.
			true_image_size (tuple): original (width x height) of image
			model_image_size (tuple): model (width x height) which is the input dimension of the model and used to rescale the coordinates
			num_cls (int): Number of classes the model need to detect
			grid_shape (tuple): grid for YOLO detection of shape (g x g)
			anchors (int): number of anchors per cell (= g*g)
		
		Returns:
		--------
			target (dict[str]: numpy.ndarray): 
					conf (int): GR x GC x A x 1, 
					box (float): GR x GCx A x 1, 
					class (int): GR x GCx A x num_cls } 
					Note: that this method only support for one anchor

		Note: GR x GC is grid-rows and grid-columns
		# TODO: only support for oner anchor per cell
	"""

	assert anchors == 1, 'Currently this method only support for <one> anchor per cell.'

	rows, cols = grid_shape
	target = { 'conf': np.zeros((rows, cols, anchors, 1), dtype=np.int64),
			   'box': np.zeros((rows, cols, anchors, 4), dtype=np.float),
			   'label': np.zeros((rows, cols, anchors, num_cls), dtype=np.int64), 
			 }

	
	if len(coord_labels) == 0:
		return target

	width, height = model_image_size[0], model_image_size[1]

	cell_width = width / cols
	cell_height = height / rows
	# msg = '[Cell] width = {}, height = {}'.format(cell_width, cell_height)
	# logger.info(msg); print(msg)

	# Since number of anchor here is only one
	for obj in coord_labels:

		obj_coord = np.array(obj[:4]).reshape(2, 2) * [width / true_image_size[0], height / true_image_size[1]]
		label = obj[4]
		# print('New coordinate = ', obj_coord, 'label = ', label)


		# Object center co-ordinates
		obj_cx, obj_cy = (obj_coord[0, 0] + obj_coord[1, 0]) / 2, (obj_coord[0, 1] + obj_coord[1,1]) / 2
		# Object width and height
		obj_w, obj_h = abs(obj_coord[1, 0] - obj_coord[0, 0]), abs(obj_coord[1, 1] - obj_coord[0, 1])
		# Object index in the 2D cell
		obj_i, obj_j = int(obj_cx // cell_width), int(obj_cy // cell_height)

		# Object-center's cell-center coordinates
		cell_cx, cell_cy = ((cell_width * obj_i) + cell_width / 2), ((cell_height * obj_j) + cell_height / 2) 

		# TODO: In case more than one anchors then use IOU to decide which anchor to select
		anchor = 0

		# In case object is already there are given index then randomly check if it will replaced or old one will exist.
		if target['conf'][obj_i, obj_j, anchor, 0] == 1:
			picker = np.random.randint(2)
			if not picker:
				continue # 

		# Compute delta center and size
		delta_cx,  delta_cy = delta_obj_center((obj_cx, obj_cy), (cell_cx, cell_cy), (cell_width, cell_height)) 		
		delta_w, delta_h = delta_obj_size((obj_w, obj_h), (width, height)) 

		# obj_cx, obj_cy = inverse_delta_obj_center((delta_cx, delta_cy), (cell_cx, cell_cy), (cell_width, cell_height))
		# obj_w, obj_h = inverse_delta_obj_size((delta_w, delta_h), (width, height))


		target['conf'][obj_i, obj_j, anchor, 0] = 1
		target['box'][obj_i, obj_j, anchor, :] = [delta_cx, delta_cy, delta_w, delta_h]
		
		# print('label = {}\nobj={}, \n([delta_cx, delta_cy, delta_w, delta_h]) = {}'.format(label, obj, ([delta_cx, delta_cy, delta_w, delta_h])))
	

		target['label'][obj_i, obj_j, anchor, label] = 1 # only given object is set to 1 others are zero


	# Note that we can store of different types in ndarray or tensor that is why passed as dictinary 
	return target


def yolo_boxes_to_corners(delta_box_xy, delta_box_wh,  model_image_size, true_image_size=None):

	"""Compute corners of the the all the detected objects

		Parameters:
		-----------
			delta_box_xy (numpy.ndarray): shape (grid_rows, grid_cols, 1, 2)
			delta_box_wh (numpy.ndarray): shape (grid_rows, grid_cols, 1, 2)
			model_image_size (tuple): width and height which is the input dimension to the model 
			true_image_size (tuple): optional- true width and height of the image which is used to rescale the coordinates if given  

		Returns:
		--------
			boxes (numpy.ndarray): shape (grid_rows, grid_cols, 1, 4)
				   where the last dimension is corners of the box

		# TODO: only support for oner anchor per cell
	"""

	rows, cols, anchors, _ = delta_box_xy.shape
	assert anchors == 1, 'Currently this method only support for <one> anchor per cell.'
	
	width, height = model_image_size 
	cell_width = width / cols
	cell_height = height / rows

	# pdb.set_trace()	
	boxes = np.zeros((rows, cols, anchors, 4))
	for i in range(rows):
		for j in range(cols):
			cell_cx, cell_cy = float((cell_width * i) + cell_width / 2), float((cell_height * j) + cell_height / 2) 

			anchor = 0
			delta_cx, delta_cy = delta_box_xy[i, j, anchor, :]
			delta_w, delta_h = delta_box_wh[i, j, anchor, :]

			obj_cx, obj_cy = inverse_delta_obj_center((delta_cx, delta_cy), (cell_cx, cell_cy), (cell_width, cell_height))
			obj_w, obj_h = inverse_delta_obj_size((delta_w, delta_h), (width, height))

			x1, y1, x2, y2 = box_coordinates((obj_cx, obj_cy), (obj_w, obj_h), (width, height))
			obj_coord = np.array([x1, y1, x2, y2]).reshape(2, 2)

			# pdb.set_trace()
			if true_image_size is not None:
				obj_coord = obj_coord * [true_image_size[0] / width, true_image_size[1] / height]

			x1, y1, x2, y2  = obj_coord.reshape(-1)
			boxes[i, j, anchor, :] = x1, y1, x2, y2
			# pdb.set_trace()	

	# pdb.set_trace()	
	return boxes


def yolo_eval(yolo_outputs, model_image_size, true_image_size=None, max_boxes=9, score_threshold=.6, iou_threshold=.5, on_true=False):
	"""Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
	
		Arguments:
			yolo_outputs -- output of the encoding model (for image_shape of (model_width, model_height, 3)), contains 4 tensors:
							box_confidence: tensor of shape (GR x GC x A x 1) => (4, 4, 1, 1)
							box_xy: tensor of shape (GR x GC x A x 2) = > (4, 4, 1, 2)
							box_wh: tensor of shape (GR x GC x A x 2) => (4, 4, 1, 2)
							box_class_probs: tensor of shape (GR x GC x A x C) => (4, 4, 1, 2)
			model_image_size (tuple): width and height which is the input dimension to the model 
			true_image_size (tuple): optional- true width and height of the image which is used to rescale the 
									 coordinates if given else it according to model size 
			max_boxes (int): maximum number of predicted boxes you'd like
			score_threshold (float): if [conf * class_prob < threshold], then get rid of the corresponding box
			iou_threshold (float) "intersection over union" threshold used for NMS filtering
			on_true (bool): if the yolo_outputs is target then no need to compute sigmoid on conf and class_prob

		Returns:
		--------
			scores (numpy.ndarray): shape (None, ), predicted score for each box
			boxes (numpy.ndarray): shape (None, 4), predicted box coordinates
			classes (numpy.ndarray): shape (None,), predicted class for each box
	
		Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
	"""
	
	### START CODE HERE ### 
	
	# Retrieve outputs of the YOLO model (≈1 line)
	print('*********************** [In] yolo_eval ***********************')
	if type(yolo_outputs) == np.ndarray:
		box_confidence = yolo_outputs[..., 0:1]
		delta_box_xy, delta_box_wh = yolo_outputs[..., 1:3], yolo_outputs[..., 3:5]
		box_class_probs = yolo_outputs[..., 5:yolo_outputs.shape[-1]]
	else: # type(yolo_outputs) == tuple
		box_confidence, delta_box, box_class_probs = yolo_outputs
		delta_box_xy, delta_box_wh = delta_box[..., 0:2], delta_box[..., 2:4]


	# TODO: uncomment
	# pdb.set_trace()
	if not on_true:
		box_confidence = torch.tensor(box_confidence).float().sigmoid().numpy()
		box_class_probs = torch.tensor(box_class_probs).float().sigmoid().numpy()


	# Convert boxes to be ready for filtering functions  and NOTE: that coordinates are transformed back which is according to original size 
	boxes = yolo_boxes_to_corners(delta_box_xy, delta_box_wh, model_image_size, true_image_size)


	# pdb.set_trace()
	# Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
	# print(box_confidence, '\n\n\n\n\n', box_class_probs)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
	

	# Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
	
	### END CODE HERE ###
	return scores, boxes, classes













