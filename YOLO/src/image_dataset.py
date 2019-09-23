# -*- coding: utf-8 -*-


import pdb
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb

import torch
from torch.utils.data import Dataset

from .transformers import NormalizeImageData, MirrorImage, InvertVerticallyImage, MirrorAndInvertVerticallyImage, Rotate90Image, Rotate270Image, RandomColorShifter 
from . import yolo_utils as yutil


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
			transformers (list[transformer]): List of transformer from which randomly one is picked for data augmentation  
			test_transformer_flag (int): A checker to control data-augmentation which should be set to (1) for unittest to test the transformers.  
	"""
	
	def __init__(self, datatype_dir, get_fnames_method, model_img_size, boxes_info_dict=None, transformers=None, test_transformer_flag=0):
		self.datatype_dir = datatype_dir
		
		self.fnames = get_fnames_method()
		self.model_image_size = model_img_size
		self.boxes_info_dict = boxes_info_dict

		self.transformers = transformers
		self.transformer_len = 0

		self.test_transformer_flag = test_transformer_flag # If this class is instantiated to perform unittest else the behaviour is normal
		if self.transformers is not None:
			self.transformer_len = len(self.transformers)

		self.normalizer = NormalizeImageData()

		


	
	def __len__(self):
		return len(self.fnames)

	def get_fnames(self):
		return self.fnames
	
	def __getitem__(self, idx):
		
		img_name = self.fnames[idx]
		# print('1:img_name = ', img_name)


		img_name_path = os.path.join(self.datatype_dir, img_name)
		
		# img_ndarray: np.ndarray of shape (H x W x C) without normalization meaning values in [0, 255]
		true_image_size, img_ndarray = yutil.load_and_resize_image(img_name_path, self.model_image_size) 

		if len(img_ndarray.shape) == 3:
			if img_ndarray.shape[2] == 4:
				img_ndarray = rgba2rgb(img_ndarray)
		else:  # If b&w image repeat the dimension
			img_ndarray = np.expand_dims(img_ndarray, axis=2)
			img_ndarray = np.concatenate((img_ndarray, img_ndarray, img_ndarray), axis=2)



		# TODO: randomly choose one of the transform for data augmentation
		# if self.transformer:
		# 	img_ndarray = self.transformer(img_ndarray)
		
		# Transpose the dimensions	
		# img_ndarray = img_ndarray.transpose(2, 0, 1) # (H x W x C) => (C x H x W)

		if self.boxes_info_dict is not None:
			boxes_info =  self.boxes_info_dict.get(img_name, [])
			# pdb.set_trace()
			

			if self.test_transformer_flag: 
				use_transformer_flag = 1 # while testing make sure to use transformer since we are testing it.
			else:
				use_transformer_flag = np.random.randint(2) # This will allow only half the times images to be transformed at others it is passed as normal


			# pdb.set_trace()
			if self.transformer_len > 0 and use_transformer_flag:
				tfr_rand_key = np.random.randint(self.transformer_len)
				# print('#tfr_rand_key = ', tfr_rand_key)
				y_tfr_func = self.transformers[tfr_rand_key]['y_tfr_func']

				target = yutil.construct_target(boxes_info, true_image_size, self.model_image_size, yutil.get_num_classes(), yutil.get_grid_shape(), yutil.get_num_anchors(), y_tfr_func=y_tfr_func)
				img_ndarray, target = self.transformers[tfr_rand_key]['x_tfr'](img_ndarray, target)
			else:
				target = yutil.construct_target(boxes_info, true_image_size, self.model_image_size, yutil.get_num_classes(), yutil.get_grid_shape(), yutil.get_num_anchors())

			# pdb.set_trace()
			img_ndarray = self.normalizer(img_ndarray)
			
			img_ndarray = img_ndarray.transpose(2, 0, 1) # (H x W x C) => (C x H x W)

			return img_ndarray, target, np.array(true_image_size)
		else:
			img_ndarray = img_ndarray.transpose(2, 0, 1) # (H x W x C) => (C x H x W)
			return img_ndarray, np.array([0]), np.array(true_image_size)


