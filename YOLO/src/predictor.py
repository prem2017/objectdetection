# -*- coding: utf-8 -*-


""" ©Prem Prakash
	Predictor module 
"""

import pdb
import os
import sys
from copy import deepcopy
import argparse

import scipy
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib.pyplot import imshow


from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss_functions import YoloLoss
from .models import AgriNet
from .image_dataset import ImageDataset

from . import yolo_utils as yutil
from . import util
from .util import logger

def load_trained_model(model_fname):
	"""Loads the pretrained model for the given model name.

		Parameters:
		-----------
			 model_fname (str): the filename of the model

		Returns:
		--------
			model (dict): {key (str): val (torch.nn.Module)} since the model was stored in key-value pair

	"""
	model_path = os.path.join(util.get_models_dir_old(), model_fname)
	
	model = {}
	
	# TODO: change it to AgriNet
	model['yolo'] = AgriNet() # # None)

	# print('\n [Before] model[regressor] = ', model['regressor'])
	
	saved_state_dict = torch.load(model_path, map_location= lambda storage, loc: storage)
	pretrained_yolo_model_state_dict = saved_state_dict['yolo']


	# Available dict
	raw_model_dict = model['yolo'].state_dict()
	
	# Extract (k, v) pair which is present in both; remaining will be randomly intialized
	# pretrained_yolo_model_state_dict = {k: v for k, v in pretrained_yolo_model_state_dict.items() if k in raw_model_dict}
	# raw_model_dict.update(pretrained_yolo_model_state_dict)

	model['yolo'].load_state_dict(pretrained_yolo_model_state_dict)
	model = {k: v.eval() for k, v in model.items()}

	# print('\n [After] update model[regressor] = ', model['regressor'])
	return model
	
		

def compute_label_and_prob(pred: torch.Tensor, th=0.5):
	"""Computes label/class and probability of it from give predcition score. Note that is only used for binary class. 
		
		Parameters:
		-----------
			pred (torch.Tensor): the predicted score
			th (float): the threshold value which decides the class

		Returns:
		--------
			label (torch.Tensor): the label from the predicted score
			 pred_prob (torch.Tensor): probability computed using sigmoid
	"""
	pred_prob = pred.sigmoid()
	
	label = deepcopy(pred)
	label[label >= th] = 1
	label[label < th] = 0

	return label,  pred_prob


def gen_conf_and_cls_report(ytrue, ypred):
	"""Generated report from the predcition such as classification-report, confusion-matrix, F1-score, Average Precision Score (aps)
	   	
	   	Parameters:
	   	-----------
	   		ytrue (torch.Tensor): the true labels for which predicted score is given
	   		ypred (torch.Tensor): the predicted value from the network

	   	Returns:
	   	--------
	   		report (dict): {key (str): val (metric-score)} key-value pair for difference metric score such as f1-score, aps etc.
	   		f1_checker (list): the f1-score of object-confidence and of different classes 
	   		report['aps'] (float): although already present in the <report> but also returned separately for saving a model in this condition as content of dict can change in future.  
	"""

	true_conf_lable = ytrue['conf'].contiguous().view(-1) # [..., 0:1]
	# true_box = ytrue['box'] # [..., 1:5]
	true_class_labels = ytrue['label'] # [..., 5:y_target.shape[-1]]


	pred_conf_val = ypred[..., 0:1].contiguous().view(-1)
	# pred_box = ypred[..., 1:5]
	pred_class_val = ypred[..., 5:ypred.shape[-1]]

	f1_checker = []

	# pdb.set_trace()
	report = {'conf': {}, 'class': {}}
	pred_conf_label, pred_conf_prob = compute_label_and_prob(pred_conf_val)

	# print('\n\ntype(y_true) = {} type(pred_conf_label) = {}'.format(true_conf_lable.type(), pred_conf_label.type()))
	report['conf']['f1'] = f1_score(y_true=true_conf_lable.cpu().numpy(), y_pred=pred_conf_label.cpu().numpy())
	f1_checker.append(report['conf']['f1'] )

	report['conf']['clf_report'] = classification_report(y_true=true_conf_lable.cpu().numpy(), y_pred=pred_conf_label.cpu().numpy())

	class_names = yutil.get_class_names()
	num_classes = len(class_names)
	apss = []
	# TODO: Average precision score should also be computed where if IOU < 0.5 then the prediction is considered false positive
	# Which is also false negative
	for i in range(num_classes):
		report['class'][class_names[i]]  = {}
		true_class_label = true_class_labels[..., i].contiguous().view(-1)
		
		pred_class_label, pred_class_prob = compute_label_and_prob(pred_class_val[..., i].contiguous().view(-1))
		
		report['class'][class_names[i]]['f1'] = f1_score(y_true=true_class_label.cpu().numpy(), y_pred=pred_class_label.cpu().numpy())
		report['class'][class_names[i]]['aps'] = average_precision_score(y_true=true_class_label.cpu().numpy(), y_score=pred_class_prob.cpu().numpy())
		
		apss.append(report['class'][class_names[i]]['aps'])
		
		report['class'][class_names[i]]['clf_report'] = classification_report(y_true=true_class_label.cpu().numpy(), y_pred=pred_class_label.cpu().numpy())
		report['class'][class_names[i]]['conf_mat'] = confusion_matrix(y_true=true_class_label.cpu().numpy(), y_pred=pred_class_label.cpu().numpy())
		
		f1_checker.append(report['class'][class_names[i]]['f1'])

	report['aps'] = sum(apss) / len(apss)
	return report, f1_checker, report['aps'] 




def draw_and_save(data_img_dir, img_fname, output, output_type, results_path):
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

	out_scores, out_boxes, out_classes = output

	# Print predictions info
	class_names = yutil.get_class_names()
	colors = yutil.get_class_colors() # generate_colors(class_names)

	# Draw bounding boxes on the image file
	yutil.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	# Save the predicted bounding box on the image
	save_path = os.path.join(results_path, output_type + '_' + img_fname)

	image.save(save_path, quality=90)
	
	# TODO: only for checking Display the results in the notebook
	output_image = scipy.misc.imread(os.path.join(results_path, output_type + '_' + img_fname))
	imshow(output_image)
	return 









def predict_on_test(models, data_info, weights=None, has_target=False, results_path=util.get_results_dir()):
	"""Predict on the given data and save the results. If target is available then also generates reports.

		Parameters:
		-----------
			models (list[dict]): list[{key (str): value (nn.Module)}] list of models or prediction where each dict could be with architecture like encoder/decoder thus key-value pair.
								 The list of model is there for prediction from ensemble of models

			data_info (dict): information needed to setup test dataset.
			weights (list): weights associated with each models
			has_target (bool: if true the method also generates report
			results_path (str): fullpath where results are genrated reports are saved
	"""

	models_len = len(models)
	if weights is None:
		weights = [1.0/models_len for i in range(models_len)]

	data_img_dir = data_info['datatype_dir']
	data_info['model_img_size'] = yutil.get_model_img_size()
	if has_target:
		data_info['boxes_info_dict'] = util.PickleHandler.extract_from_pickle(util.get_obj_coord_pickle_datapath())
	dataset = ImageDataset(**data_info)
	img_fname_all = dataset.get_fnames()
	test_dataloader = DataLoader(dataset=dataset, batch_size=util.get_test_batch_size(), shuffle=False)

	msg = '[Predict] data_info = {}'.format(data_info)
	logger.info(msg)

	# For now there is only one model
	models = [{k: v.to(util.device).eval() for k, v in model.items()} for model in models]



	y_pred_all = None
	y_true_all = None
	true_image_size_all = None
	with torch.no_grad():
		for i, (x, y, true_image_size) in enumerate(test_dataloader):
			print('[val] i = ', i)
			loss_val = 0
			
			x = x.to(device=util.device, dtype=torch.float)
			y_pred = None
			for i, model in enumerate(models):
				output = model['yolo'](x)
				output = weights[i] * output 

				y_pred = output if y_pred is None else (y_pred + output)



			y_pred_all = y_pred if y_pred_all is None else torch.cat((y_pred_all, y_pred), dim=0)
			print('[Inside] y_pred_all.shape = ', y_pred_all.shape)
			if has_target and type(y) == dict: 
				if y_true_all is None:
					y_true_all = {}
					for k, v in y.items():
						print('k = {}, v.sum() = {} '.format(k, v.sum()))
						y_true_all[k] = v
				else:
					for k, v in y.items():
						print('k = {}, v.sum() = {} '.format(k, v.sum()))
						y_true_all[k] = torch.cat((y_true_all[k], v), dim=0)
						print('k = {}, y_true_all[k].sum() = {} '.format(k, y_true_all[k].sum()))

			true_image_size_all = true_image_size if true_image_size_all is None else torch.cat((true_image_size_all, true_image_size), dim=0)
	# End of inference
	
	if has_target:
		test_report, f1_checker, aps = gen_conf_and_cls_report(y_true_all, y_pred_all)
		msg = '[Test] report =\n {}'.format(util.pretty(test_report))
		logger.info(msg); print(msg)


	model_image_size = yutil.get_model_img_size()
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	msg = '[Results] stored at path = {}'.format(results_path)
	logger.info(msg); print(msg)

	output = None
	for i in range(true_image_size_all.shape[0]):
		true_image_size = true_image_size_all[i] # .cpu().numpy()
		true_image_size = tuple(true_image_size.cpu().numpy())

		y_pred = y_pred_all[i]
		y_pred = y_pred.cpu().numpy()

		box_confidence = y_pred[..., 0:1]
		delta_box = y_pred[..., 1:5]
		box_class_probs = y_pred[..., 5:y_pred.shape[-1]]

		yolo_outputs = (box_confidence, delta_box, box_class_probs)
		# msg = '-'.join([str(type(val)) for val in yolo_outputs])
		# print(msg)

		img_fname = img_fname_all[i]
		print('\n\n *************************** Image Name = {} *************************** \n\n'.format(img_fname))
		# pdb.set_trace()
		pred_scores, pred_boxes, pred_classes = yutil.yolo_eval(yolo_outputs=yolo_outputs, model_image_size=model_image_size, true_image_size=true_image_size, max_boxes=9, score_threshold=.45, iou_threshold=.6)
		

		# pdb.set_trace()
		output = pred_scores, pred_boxes, pred_classes
		
		draw_and_save(data_img_dir=data_img_dir, img_fname=img_fname, output=output, output_type='pred', results_path=results_path)
		
		print('Done with predicted now True')
		if y_true_all is not None and has_target:
			
			# pdb.set_trace()
			
			y_true = []
			for k, v in y_true_all.items():
				y_true.append(v[i].cpu().numpy())

			true_scores, true_boxes, true_classes = yutil.yolo_eval(yolo_outputs=tuple(y_true), model_image_size=true_image_size, true_image_size=true_image_size, max_boxes=9, score_threshold=.6, iou_threshold=.5, on_true=True)
			output = true_scores, true_boxes, true_classes
			draw_and_save(data_img_dir=data_img_dir, img_fname=img_fname, output=output, output_type='true', results_path=results_path)




def get_arguments_parser(default_test_datapath, default_results_path):
	"""Argument parser for predition"""
	description = 	'Provide arguments for fullpath to directory where \n' \
					'**Test** files are located and fullpath to \n' \
					'where **Results** will be saved and also indicate if data has target value to generate report.'
					
	parser = argparse.ArgumentParser(description=description)


	parser.add_argument('-t', '--test', type=str, default=default_test_datapath,
						 help='Provide fullpath to where the test images are stored.', required=False)


	parser.add_argument('-r', '--resultspath', type=str, default=default_results_path,
						 help='Provide fullpath to where results should be stored', required=False)

	parser.add_argument('--has_target', type=bool, default=False,
						 help='Indicates if the test data has target (bool).', required=False)

	return parser


			

if __name__ == '__main__':
	print('[Run Test]')
	
	util.reset_logger('predictor_output.log')

	# First set the model_name and load 
	rows, cols = yutil.get_grid_shape()
	anchors = yutil.get_num_anchors()
	util.set_trained_model_name(rows, cols, anchors)
	model_fname = util.get_trained_model_name()
	print(model_fname)


	# extra = 'epochs_swa_ensemble'
	# extensions = ['', '_mintrain', '_minval', '_maxconf_f1', '_maxaps']
	# for ex in extensions:
	ex =  '_maxaps' # '_maxconf_f1'# '_mintrain' '_minval' _maxaps
	model_fnames = [model_fname + ex] # _maxconf_f1,   _minval model_fname + '_mintrain', model_fname + '_minval', model_fname + '_min_mae_val']
	msg = '#########################################################################'
	msg += '\n##  {}  ##'.format(model_fnames)
	msg += '\n#########################################################################'
	logger.info(msg); print(msg)



	results_dir_name = '_'.join(model_fnames)
	results_dir = os.path.join(os.path.basename(util.get_models_dir_old()), results_dir_name)
	default_results_path = util.get_results_dir(results_dir)
	print(model_fnames)

	default_test_datapath = util.get_test_datapath()
	arg_parser = get_arguments_parser(default_test_datapath, default_results_path)
	arg_parser = arg_parser.parse_args()
	test_datapath = arg_parser.test
	results_path = arg_parser.resultspath
	has_target = arg_parser.has_target
	msg = '[Args]: \ntest_datapath = {}, \nresults_path = {}, has_target={}'.format(test_datapath, results_path, has_target)
	logger.info(msg); print(msg)



	msg = '[Model] names = {}'.format(model_fnames)
	logger.info(msg); print(msg)
	models = [load_trained_model(model_fname) for model_fname in model_fnames]
	
	

	data_info = {'datatype_dir': test_datapath, 'get_fnames_method': util.get_test_fnames} 
	msg = '[Filepath] Test data info = {}'.format(data_info)
	logger.info(msg); print(msg)


	default_weights = [1] # [0.55, 0.25, 0.10, .10] # [1] # when only one model
	# TODO: has_target_flag
	predict_on_test(models=models, data_info=data_info, weights=default_weights, has_target=True, results_path=results_path)



	msg = "#------------------------------ Prediction Completed ------------------------------#"
	logger.info(msg); print(msg)







