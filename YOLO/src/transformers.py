# -*- coding: utf-8 -*-

""" ©Prem Prakash
	Transformers module 

	#TODO: For further data augmentation flip, change in HSL value
"""

import pdb
import os

import torch
import torchvision.transforms as torch_transformer

import numpy as np
import skimage.transform as sk_transformer
import cv2




################ Series of Transformers  ################


# y is target['conf']
# target['box']
# target['label']

# https://scottontechnology.com/flip-image-opencv-python/
# https://www.science-emergence.com/Articles/How-to-flip-an-image-horizontally-in-python-/
# # flip img horizontally, vertically,
# and both axes with flip()
# if imgage shape is of HxWxC form i.e. height, width, channel the  at 
# horizontal_img = cv2.flip( img, 0 ) # 
# vertical_img = cv2.flip( img, 1 )
# both_img = cv2.flip( img, -1 )
class MirrorImage(object):
	"""Horizontal flip of image where the input array is numpy.ndarray and of form H x W x C

	"""
	def __init__(self):
		super(MirrorImage, self).__init__()
		


	@staticmethod
	def mirror_align_objects_coord(coordinates, image_size):
		"""
			Parameters:
			-----------
				coordinates (numpy.ndarray): [[x1, y1], [x2, y2]] of object
				model_width (int): width of image that is fed in the images

			Returns:
			--------
				coordinates (numpy.ndarray): [[model_width - x2, y1], [model_width - x1, y2]] since righmost becomes leftmost
		"""
		width, height = image_size
		coordinates[::-1, 0] = width - coordinates[:, 0]
		return coordinates

	
	@staticmethod
	def transform(x):
		return np.flip(x, axis=1)


	def __call__(self, x, y):
		"""Take mirror image and object coordinates are transformed while constructing target

			Parameters:
			-----------
				x (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				x (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		x = np.flip(x, axis=1).copy() # because width is at second dimension i.e. axis=1
		
		# y_conf, y_box, y_label = y['conf'], y['box'], y['label']
		# pdb.set_trace()
		# for key in y.keys():
		# 	y[key] = np.flip(y[key], axis=1).copy()

		return x, y


class InvertVerticallyImage(object):
	"""Vertical flip of image where the input array is numpy.ndarray and of form H x W x C

	"""
	def __init__(self):
		super(InvertVerticallyImage, self).__init__()


	@staticmethod
	def invert_vetically_objects_coord(coordinates, image_size):
		"""
			Parameters:
			-----------
				coordinates (numpy.ndarray): [[x1, y1], [x2, y2]] of object
				model_width (int): width of image that is fed in the images

			Returns:
			--------
				coordinates (numpy.ndarray): [[x1, height - y2], [x2, height - y1]] since top becomes bottom
		"""
		width, height = image_size
		coordinates[::-1, 1] = height - coordinates[:, 1]
		return coordinates

	
	@staticmethod
	def transform(x):
		return np.flip(x, axis=0)


	def __call__(self, x, y):
		"""Invert image vertically and object coordinates are transformed while constructing target

			Parameters:
			-----------
				x (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				x (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		x = np.flip(x, axis=0).copy() # because width is at second dimension i.e. axis=1
		return x, y


class MirrorAndInvertVerticallyImage(object):
	"""Mirror/Horizontal and Vertical flip of image where the input array is numpy.ndarray and of form H x W x C """
	
	def __init__(self):
		super(MirrorAndInvertVerticallyImage, self).__init__()


	@staticmethod
	def mirror_and_invert_vetically_objects_coord(coordinates, image_size):
		"""
			Parameters:
			-----------
				coordinates (numpy.ndarray): [[x1, y1], [x2, y2]] of object
				model_width (int): width of image that is fed in the images

			Returns:
			--------
				coordinates (numpy.ndarray): [[width - x2, height - y2], [width - x1, height - y1]] since left becomes right and top becomes bottom
		"""
		width, height = image_size
		coordinates[::-1, 0] = width - coordinates[:, 0]
		coordinates[::-1, 1] = height - coordinates[:, 1]
		return coordinates

	
	@staticmethod
	def transform(x):
		return np.flip(np.flip(x, axis=1), axis=0)


	def __call__(self, x, y):
		"""Take mirror image and also invert vetically and object coordinates are transformed while constructing target

			Parameters:
			-----------
				x (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				x (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		x = np.flip(np.flip(x, axis=1), axis=0).copy() # because width is at second dimension i.e. axis=1
		return x, y


class Rotate90Image(object):
	"""Rotate image by 90 degree in clockwise (axes=(1,0)) the input array is numpy.ndarray and of form H x W x C
	    
	   Note: For now it only works if image is square else after rotation the image size and coordinated need to be recalibrated
	"""

	def __init__(self):
		super(Rotate90Image, self).__init__()
		
	@staticmethod
	def rotate_objects_coord_90(coordinates, image_size):
		"""
			Parameters:
			-----------
				coordinates (numpy.ndarray): [[x1, y1], [x2, y2]] of object
				model_width (int): width of image that is fed in the images

			Returns:
			--------
				coordinates (numpy.ndarray): [[model_width - x2, y1], [model_width - x1, y2]] since righmost becomes leftmost
		"""
		width, height = image_size
		[[x1, y1], [x2, y2]]  = coordinates

		coordinates = np.array([[height - y2, x1], [height - y1, x2]])
		return coordinates

	
	@staticmethod
	def transform(x):
		return np.rot90(x, k=1, axes=(1, 0)) 


	def __call__(self, x, y):
		"""Rotate image by 270 degree and object coordinates are transformed while constructing target

			Parameters:
			-----------
				x (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				x (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		x = np.rot90(x, k=1, axes=(1, 0)).copy() # because width is at second dimension i.e. axis=1
		return x, y
		


class Rotate270Image(object):
	"""Rotate image by 270 degree in clockwise (axes=(1,0)) the input array is numpy.ndarray and of form H x W x C
	    
	   Note: For now it only works if image is square else after rotation the image size and coordinated need to be recalibrated
	"""
	def __init__(self):
		super(Rotate270Image, self).__init__()
		
	@staticmethod
	def rotate_objects_coord_270(coordinates, image_size):
		"""
			Parameters:
			-----------
				coordinates (numpy.ndarray): [[x1, y1], [x2, y2]] of object
				model_width (int): width of image that is fed in the images

			Returns:
			--------
				coordinates (numpy.ndarray): [[model_width - x2, y1], [model_width - x1, y2]] since righmost becomes leftmost
		"""
		width, height = image_size
		[[x1, y1], [x2, y2]]  = coordinates

		coordinates = np.array([[y1, width - x2], [y2, width - x1]])
		return coordinates

	
	@staticmethod
	def transform(x):
		return np.rot90(x, k=3, axes=(1, 0)) 


	def __call__(self, x, y):
		"""Rotate image by 270 degree and object coordinates are transformed while constructing target

			Parameters:
			-----------
				x (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				x (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		x = np.rot90(x, k=3, axes=(1, 0)).copy() # because width is at second dimension i.e. axis=1
		return x, y
		



class ChangeHueImage(object):
	"""docstring for ChangeHueImage"""
	def __init__(self, arg):
		super(ChangeHueImage, self).__init__()
		self.arg = arg
		

class ChangeSaturation(object):
	"""docstring for ChangeSaturation"""
	def __init__(self, arg):
		super(ChangeSaturation, self).__init__()
		self.arg = arg


class ChangeLuminescenceImage(object):
	"""docstring for ChangeLuminescenceImage"""
	def __init__(self, arg):
		super(ChangeLuminescenceImage, self).__init__()
		self.arg = arg


# rgb_randomize
# Brightness 
# contrast		
		


# Rescale/Resize the with the given dim
class RescaleImage(object):
	"""RescaleImage the given for the given dimension

	Parameters:
	-----------
		output_size (tuple): (height width)
	"""
	
	def __init__(self, output_size):
		self.output_size = output_size
	
	def __call__(self, image_ndarray):
		new_h, new_w = self.output_size
		
		new_img = sk_transformer.resize(image_ndarray, (new_h, new_w))
		return new_img


class NormalizeImageData(object):
	""" Normalize the image data per channel for quicker and better training """

	def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
		"""Default values are from imagenet"""
		self.means = torch.tensor(means)
		self.stds = torch.tensor(stds)
	
	def __call__(self, img_ndarray):
		""" Normalize the image data

			Parameters:
			-----------
				img_ndarray (numpy.ndarray): (H x W x C)

			Returns:
				img_ndarray (torch.Tensor): normalized image of type torch.Tensor

		"""
		assert len(
			img_tensor.shape) == 3, '[Assertion Error]: Normalization is performed for each channel so input must be only one image data only'
		img_tensor = torch.tensor(img_ndarray)
		img_tensor = img_tensor.contiguous()

		# If normalization is done idividually 
		# mean_list = img_tensor.view(img_tensor.shape[0], -1).mean(dim=1)
		# std_list = img_tensor.view(img_tensor.shape[0], -1).std(dim=1)
		
		normalizer = torch_transformer.Normalize(mean=self.means, std=self.stds)
		return normalizer(img_tensor)






