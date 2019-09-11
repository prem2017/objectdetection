#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Prem Prakash
# Transformers module tests cases

__author__ = 'Prem Prakash'


import os
import pdb
import sys
import unittest

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

# TODO: Import only what is needed 
from ..src.image_dataset import ImageDataset
from ..src.transformers import MirrorImage, InvertVerticallyImage, MirrorAndInvertVerticallyImage, Rotate90Image, Rotate270Image

# from ..src.predictor import *

from ..src import util
from ..src import yolo_utils as yutil

print('Hello Tests \n\n')

# Use decorator to disable 
# https://stackoverflow.com/questions/2066508/disable-individual-python-unit-tests-temporarily/#2066729
def disabled(f):
    def _decorator():
        print(f.__name__ + ' has been disabled')
    return _decorator

class TestTransformers(unittest.TestCase):
	"""docstring for TestTransformers"""
	def __init__(self, *args, **kwargs):
		super(TestTransformers, self).__init__( *args, **kwargs)

		# def __init__(self, datatype_dir, get_fnames_method, model_img_size, boxes_info_dict=None, transformers=None):
		test_datapath = util.get_test_datapath()
		data_info = {}
		data_info['datatype_dir'] = test_datapath

		data_info['get_fnames_method'] = util.get_test_fnames
		data_info['model_img_size'] = yutil.get_model_img_size()
		data_info['boxes_info_dict'] = util.PickleHandler.extract_from_pickle(util.get_obj_coord_pickle_datapath())

		self.data_info = data_info
		self.data_img_dir = data_info['datatype_dir']


	@unittest.SkipTest
	def test_mirror_image_transformer(self):

		self.data_info['transformers'] = {0: {'x_tfr': MirrorImage(), 'y_tfr_func': MirrorImage.mirror_align_objects_coord}}
		tfr_func = MirrorImage.transform

		results_path = util.get_results_dir('mirror_image_tests/')
		output_type = 'true_mirrored'
		print('**** results_path = ', results_path)
		if not os.path.exists(results_path):
			os.makedirs(results_path)
			print('\n\n [Stored] mirrored test results are stored at path = {} \n\n'.format(results_path))

		test_common_codes(self.data_img_dir, self.data_info, output_type, results_path, tfr_func=tfr_func)
		self.assertTrue(True)

	@unittest.SkipTest
	def test_invert_vertically_image_transformer(self):

		self.data_info['transformers'] = {0: {'x_tfr': InvertVerticallyImage(), 'y_tfr_func': InvertVerticallyImage.invert_vetically_objects_coord}}
		tfr_func = InvertVerticallyImage.transform

		results_path = util.get_results_dir('invert_vertically_image_tests/')
		output_type = 'true_inverted_vertically'
		print('**** results_path = ', results_path)
		if not os.path.exists(results_path):
			os.makedirs(results_path)
			print('\n\n [Stored] inverted vertically test results are stored at path = {} \n\n'.format(results_path))
		
		test_common_codes(self.data_img_dir, self.data_info,  output_type, results_path, tfr_func=tfr_func)
		self.assertTrue(True)


	@unittest.SkipTest
	def test_mirror_and_invert_vertically_image_transformer(self):

		self.data_info['transformers'] = {0: {'x_tfr': MirrorAndInvertVerticallyImage(), 'y_tfr_func': MirrorAndInvertVerticallyImage.mirror_and_invert_vetically_objects_coord}}
		tfr_func = MirrorAndInvertVerticallyImage.transform

		results_path = util.get_results_dir('mirror_and_invert_vertically_image_tests/')
		output_type = 'true_mirrored_and_inverted_vertically'
		print('**** results_path = ', results_path)
		if not os.path.exists(results_path):
			os.makedirs(results_path)
			print('\n\n [Stored] inverted vertically test results are stored at path = {} \n\n'.format(results_path))
		
		test_common_codes(self.data_img_dir, self.data_info,  output_type, results_path, tfr_func=tfr_func)
		self.assertTrue(True)


	def test_rotate90_image_transformer(self):

		self.data_info['transformers'] = {0: {'x_tfr': Rotate90Image(), 'y_tfr_func': Rotate90Image.rotate_objects_coord_90}}
		tfr_func = Rotate90Image.transform

		results_path = util.get_results_dir('rotated90_image_tests/')
		output_type = 'true_rot90'
		print('**** results_path = ', results_path)
		if not os.path.exists(results_path):
			os.makedirs(results_path)
			print('\n\n [Stored] 90d Rotated test results are stored at path = {} \n\n'.format(results_path))
		
		test_common_codes(self.data_img_dir, self.data_info,  output_type, results_path, tfr_func=tfr_func)
		self.assertTrue(True)

	def test_rotate270_image_transformer(self):

		self.data_info['transformers'] = {0: {'x_tfr': Rotate270Image(), 'y_tfr_func': Rotate270Image.rotate_objects_coord_270}}
		tfr_func = Rotate270Image.transform

		results_path = util.get_results_dir('rotated270_image_tests/')
		output_type = 'true_rot270'
		print('**** results_path = ', results_path)
		if not os.path.exists(results_path):
			os.makedirs(results_path)
			print('\n\n [Stored] 270d Rotated test results are stored at path = {} \n\n'.format(results_path))
		
		test_common_codes(self.data_img_dir, self.data_info,  output_type, results_path, tfr_func=tfr_func)
		self.assertTrue(True)


def test_common_codes(data_img_dir, data_info, output_type, results_path, tfr_func, ):
	"""Common codes to test code and write results """

	img_dataset = ImageDataset(**data_info)
	img_fname_all = img_dataset.get_fnames()

	dataloader = DataLoader(img_dataset, batch_size=1, shuffle=False)
	
	for i, (x, y, true_image_size) in enumerate(dataloader):
	
		# pdb.set_trace()
		x = x[0].numpy()
		true_image_size = tuple(true_image_size[0].numpy())	
		img_fname = img_fname_all[i]

		y_true = []
		for k, v in y.items():
			y_true.append(v[0].numpy())

		true_scores, true_boxes, true_classes = yutil.yolo_eval(yolo_outputs=tuple(y_true), model_image_size=true_image_size, true_image_size=true_image_size, max_boxes=9, score_threshold=.6, iou_threshold=.5, on_true=True)
		output = true_scores, true_boxes, true_classes
		draw_and_save(data_img_dir=data_img_dir, img_fname=img_fname, output=output, output_type=output_type, results_path=results_path, tfr_func=tfr_func)


	
def draw_and_save(data_img_dir, img_fname, output, output_type, results_path, tfr_func=None):
	"""Draws on the images for the detected object and saves them on the given location.
	
		Parameters:
		-----------
			data_img_dir (str): directory path where images are saved 
			img_fname (str): name of an image stored in the <data_img_dir>
			output (tuple): (out_scores, out_boxes, out_classes) to draw those boxes of detected objets 
			output_type (str): in ['true', 'pred'] 
			results_path (str): directory path where images are saved after drawing
		
		Returns: 
		--------
			None
			
	"""

	# Preprocess your image
	img_path = os.path.join(data_img_dir, img_fname)
	image = Image.open(img_path)
	image = np.asarray(image)
	
	if tfr_func is not None:
		image = tfr_func(image)


	image = Image.fromarray(image)


	out_scores, out_boxes, out_classes = output

	# Print predictions info
	class_names = yutil.get_class_names()
	colors = yutil.get_class_colors() # generate_colors(class_names)

	# Draw bounding boxes on the image file
	yutil.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	# Save the predicted bounding box on the image
	save_path = os.path.join(results_path, output_type + '_' + img_fname)

	image.save(save_path, quality=90)
	return 


if __name__ == '__main__':
	unittest.main()
	print('************* Transformer Tests completer *************')



		
