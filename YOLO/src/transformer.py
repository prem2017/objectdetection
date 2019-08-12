# -*- coding: utf-8 -*-

""" ©Prem Prakash
	Transformer module 

	#TODO: For further data augmentation flip, change in HSL value
"""

import os

import torch
import torchvision.transforms as torch_transformer

import numpy as np
import skimage.transform as sk_transformer




################ Series of Transformers  ################
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






