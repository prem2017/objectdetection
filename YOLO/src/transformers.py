# -*- coding: utf-8 -*-

""" ©Prem Prakash
	Transformers module 
"""

import pdb
import os

import torch
import torchvision.transforms as torch_transformer


import numpy as np
import skimage.transform as sk_transformer
from skimage import color
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
	def transform(X):
		return np.flip(X, axis=1)


	def __call__(self, X, y):
		"""Take mirror image and object coordinates are transformed while constructing target

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		X = np.flip(X, axis=1).copy() # because width is at second dimension i.e. axis=1
		
		# y_conf, y_boX, y_label = y['conf'], y['box'], y['label']
		# pdb.set_trace()
		# for key in y.keys():
		# 	y[key] = np.flip(y[key], axis=1).copy()

		return X, y


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
	def transform(X):
		return np.flip(X, axis=0)


	def __call__(self, X, y):
		"""Invert image vertically and object coordinates are transformed while constructing target

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		X = np.flip(X, axis=0).copy() # because width is at second dimension i.e. axis=1
		return X, y


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
	def transform(X):
		return np.flip(np.flip(X, axis=1), axis=0)


	def __call__(self, X, y):
		"""Take mirror image and also invert vetically and object coordinates are transformed while constructing target

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		X = np.flip(np.flip(X, axis=1), axis=0).copy() # because width is at second dimension i.e. axis=1
		return X, y


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
	def transform(X):
		return np.rot90(X, k=1, axes=(1, 0)) 


	def __call__(self, X, y):
		"""Rotate image by 270 degree and object coordinates are transformed while constructing target

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		X = np.rot90(X, k=1, axes=(1, 0)).copy() # because width is at second dimension i.e. axis=1
		return X, y
		


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
	def transform(X):
		return np.rot90(X, k=3, axes=(1, 0)) 


	def __call__(self, X, y):
		"""Rotate image by 270 degree and object coordinates are transformed while constructing target

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray
				y (dict): no changes as already changed during target construction
		"""
		# pdb.set_trace()
		X = np.rot90(X, k=3, axes=(1, 0)).copy() # because width is at second dimension i.e. axis=1
		return X, y
	


class RandomColorShifter(object):
	"""Shift the color of image by taking average of randomly selected two of the channels (from RGB) and replacing with it, 
	   and nullify i.e. replace with zero the remaining channel
	   For example averaging Red (R) and Blue (B) and nullifying the Green (G) will make the image look purply 
	"""


	def __init__(self):
		super(RandomColorShifter, self).__init__()
		
	@staticmethod
	def transform(X):
		"""Shift the RGB gradient of image by randomly selecting one of RGB channel to set it zero and
		   average of other two.	

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
	
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with shifted colors
				
		"""
		
		# as we changing/resetting some of color gradients
		# pdb.set_trace()
		X.setflags(write=1)

		color_indices = {0, 1, 2}
		sub_index = np.random.randint(3)

		add_indices = list(color_indices - {sub_index})

		added_color_grd = np.sum(X[:, :, add_indices], axis=2) / 2
		added_color_grd = added_color_grd.astype(int)

		# pdb.set_trace()
		for id in add_indices:
			X[:, :, id] = added_color_grd

		# X = X - X[:, :, sub_index:sub_index+1]
		X[:, :, sub_index] = 0
		X[X < 0] = 0

		return X
		
	def __call__(self, X, y):
		""" Shift the RGB gradient of image 
			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				y (dict): y is key, value pair of confidence, box-coordinates, and label. 

			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with shifted colors
				y (dict): no changes needed for y since this only manipulate RGB gradient			
		"""

		return RandomColorShifter.transform(X), y



class ChangeHueImage(object):
	"""Randomly change hue of the image"""

	def __init__(self, low=-0.5, high=0.5):
		super(ChangeHueImage, self).__init__()
		self.low = low
		self.high = high
		
	@staticmethod
	def transform(X, low=-0.5, high=0.5):
		""" Change the hue of image which is selection of RGB gradient meaning which part will dominate i.e. from R(0), G(0.33), B(0.67) and then back to red	

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
	
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with changed hue
		"""

		# pdb.set_trace()
		X = X.astype(np.float32)
		hue = np.random.uniform(low, high, 1).round(2)[0]
		# print('hue = ', hue)
		hsv = color.rgb2hsv(X)
		hsv[:, :, 0] += hue
		hsv[:, :, 0] = hsv[:, :, 0].clip(min=0, max=1)
		# hsv[:, :, 1] = 1  # Turn up the saturation; we want the color to pop!
		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X, y):
		return ChangeHueImage.transform(X, self.low, self.high), y



class ChangeSaturation(object):
	"""Randomly change saturation of the image"""

	def __init__(self, low=-0.3, high=0.3):
		super(ChangeSaturation, self).__init__()
		self.low = low
		self.high = high

	@staticmethod
	def transform(X, low=-0.3, high=0.3):
		""" Change the saturation of image color which is intensity of color meaning 0: Black, 1: Full intensity of that color	

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
	
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with changed saturation
		"""

		# pdb.set_trace()
		X = X.astype(np.float32)

		saturation = np.random.uniform(low, high, 1).round(2)[0]
		# print('saturation = ', saturation)
		hsv = color.rgb2hsv(X)
		# hsv[:, :, 0] = 1
		hsv[:, :, 1] += saturation  
		hsv[:, :, 1] = hsv[:, :, 1].clip(min=0, max=1)

		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X, y):
		return ChangeSaturation.transform(X, self.low, self.high), y



class ChangeLuminescenceImage(object):
	"""docstring for ChangeLuminescenceImage"""
	def __init__(self, low=-20, high=90):
		super(ChangeLuminescenceImage, self).__init__()
		self.low = low
		self.high = high

	@staticmethod
	def transform(X, low=-20, high=90):
		""" Change the saturation of image brightness which is how bright is of colors in the image 0: Black, 1: towards white and 0.5 is original color will be retained i.e. no brightness added or removed	

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
	
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with changed luminescence/brightness
		"""

		# pdb.set_trace()
		X = X.astype(np.float32)

		luminescence = np.random.randint(low, high)
		# print('luminescence = ', luminescence)
		hsv = color.rgb2hsv(X)
		# hsv[:, :, 0] = 1
		# hsv[:, :, 1] = saturation
		# pdb.set_trace()  
		hsv[:, :, 2] += luminescence
		hsv[:, :, 2] = hsv[:, :, 2].clip(min=60, max=250)  # Normally it also between 0, 1 but becaue of <X = X.astype(np.float32)> it gets between 0 and 255

		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X, y):
		return ChangeLuminescenceImage.transform(X, self.low, self.high), y


class ChangeHSLImage(object):
	"""docstring for ChangeHSLImage"""
	def __init__(self):
		super(ChangeHSLImage, self).__init__()
		# self.low = low
		# self.high = high

	@staticmethod
	def transform(X):
		""" Randomly manipulate Hue(H), Saturation(S), Luminescence(L) of image. See change above to check how individually them impact the image 

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
	
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with changed HSL value of the image
		"""

		# pdb.set_trace()
		X = X.astype(np.float32)
		hue = np.random.uniform(-0.2, 0.6, 1).round(2)[0]
		saturation = np.random.uniform(-0.2, 0.2, 1).round(2)[0]
		luminescence = np.random.randint(-20, 90)

		# print('hue = {}, saturation= {},  luminescence = '.format(hue , saturation, luminescence))
		
		hsv = color.rgb2hsv(X)
		hsv[:, :, 0] += hue
		hsv[:, :, 1] += saturation
		hsv[:, :, 2] += luminescence

		hsv[:, :, 0:2] = hsv[:, :, 0:2].clip(min=0, max=1)  
		hsv[:, :, 2] = hsv[:, :, 2].clip(min=60, max=250)  

		X = color.hsv2rgb(hsv)
		X = X.astype(np.uint8)
		return X


	def __call__(self, X, y):
		return ChangeLuminescenceImage.transform(X), y




# rgb_randomize
# Brightness 
# contrast		


class NormalizeImageData(object):
	"""Normalize the image data per channel for quicker and better training 
	   TODO: Normalization value should computed locally for now it on Imagenet data 	
	"""

	def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
		self.means = np.array(means)
		self.stds = np.array(stds)
	
	def __call__(self, X):
		""" Normalize the image data

			Parameters:
			-----------
				X (numpy.ndarray): The RGB dimension of the image of type numpy ndarray of form H x W x C
				
			Returns:
			--------
				X (numpy.ndarray): Transformed ndarray with shifted colors
		"""

		assert len(X.shape) == 3, '[Assertion Error]: Normalization is performed for each channel so input must be only one image data only'
		# img_tensor = torch.tensor(img_ndarray)
		# img_tensor = img_tensor.contiguous()
		X = X / 255

		# If normalization is done idividually 
		# mean_list = img_tensor.view(img_tensor.shape[0], -1).mean(dim=1)
		# std_list = img_tensor.view(img_tensor.shape[0], -1).std(dim=1)
		
		# normalizer = torch_transformer.Normalize(mean=self.means, std=self.stds)
		X = (X - self.means) / self.stds 

		return X


		



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



