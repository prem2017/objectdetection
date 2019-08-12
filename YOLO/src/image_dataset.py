# -*- coding: utf-8 -*-

import os
import math

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb

import torch
from torch.utils.data import Dataset

import yolo_utils as yutil

import pdb


##
class ImageDataset(Dataset):
	""" Image dataset for custom reading of images 

		Parameters:
		----------
			data_type_dir (str): the directory path to the type of data is loaded i.e. train, test, val 
			get_fnames_method (function): to get all the filenames using this method
			model_img_size (tuple): (width, height) which is input dimension to the model
			boxes_info_dict (dict): {key (img_name: value (numpy.ndarray))} the object information of the objects.
									Available only for those which has at least object in the image.
			transformer: Data augmentation that can be applied 

			#TODO: Implement data augmentation such as flip, and manipulating hue, saturation, luminescence
	"""
	
	def __init__(self, datatype_dir, get_fnames_method, model_img_size, boxes_info_dict=None, transformer=None):
		self.datatype_dir = datatype_dir
		
		self.fnames = get_fnames_method()
		self.model_image_size = model_img_size
		self.boxes_info_dict = boxes_info_dict

		# TODO: still to use
		self.transformer = transformer

	
	def __len__(self):
		return len(self.fnames)

	def get_fnames(self):
		return self.fnames
	
	def __getitem__(self, idx):
		
		img_name = self.fnames[idx]
		# print('1:img_name = ', img_name)


		img_name_path = os.path.join(self.datatype_dir, img_name)
		
		# img_ndarray: np.ndarray of shape (H x W x C)
		true_image_size, img_ndarray = yutil.load_and_resize_image(img_name_path, self.model_image_size) 

		if len(img_ndarray.shape) == 3:
			if img_ndarray.shape[2] == 4:
				img_ndarray = rgba2rgb(img_ndarray)
		else:  # If b&w image repeat the dimension
			img_ndarray = np.expand_dims(img_ndarray, axis=2)
			img_ndarray = np.concatenate((img_ndarray, img_ndarray, img_ndarray), axis=2)



		# TODO: randomly choose one of the transform for data augmentation
		if self.transformer:
			img_ndarray = self.transformer(img_ndarray)
		
		# Transpose the dimensions	
		img_ndarray = img_ndarray.transpose(2, 1, 0) # (H x W x C) => (C x W x H)

		if self.boxes_info_dict is not None:
			boxes_info =  self.boxes_info_dict.get(img_name, [])
			# pdb.set_trace()
			target = yutil.construct_target(boxes_info, true_image_size, self.model_image_size, yutil.get_num_classes(), yutil.get_grid_shape(), yutil.get_num_anchors())
			return img_ndarray, target, np.array(true_image_size)
		else:
			return img_ndarray, np.array([0]), np.array(true_image_size)


