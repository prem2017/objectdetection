# -*- coding: utf-8 -*-


import pdb
import os
import math

import torch
import torch.nn as nn

from . import util
from .util import logger


# Loss function
class YoloLoss(nn.Module):
	"""Computes loss for  'you only look once (YOLO)' method of detecting object where the target is split into 
	   three parts:
		
		Note: the weight given to positive class is to accentuate its impact especilly in case of labels to balance out the data imbalance. 
	   	Confidence Loss (obj vs no-obj): to compute the cross entropy for confidence of object vs no-object
	   		.. math::
	   			conf_loss =  \Sigma  \left[\text{conf_pos_weight} \cdot true_y \cdot \log \sigma(pred_y) + (1 - true_y) \cdot \log (1 - \sigma(pred_y)) \right]
		
		Coordinate Loss: ((delta_x, delta_y), (delta_w, delta_y))
			.. math::
			coord_loss = \Sigma \left[	(true_dx - pred_dx)^2 +  
									   	(true_dy - pred_dy)^2 +
										(true_dw - pred_dw)^2 +
										(true_dh - pred_dh)^2
								\right]

		Class Loss: this loss also uses cross entropy but instead of multiclass it treats the problem as multilabel 
					where an anchor in cell can predict any of the class. Recall tht filtering is performed based on score which is (conf * class_prob)
					.. math::
						class_loss =  \SIgma_{c=0}^{C} \left[ \Sigma  \text{class_pos_weight[c]} \cdot true_cl[x] \cdot \log \sigma(pred_cl[c]) + (1 - true_cl[c]) \cdot \log (1 - \sigma(pred_cl[c]))  \right\]

		
		Note: The lossess computed does not reduce i.e. average because doing that will make an image with 10 object have same 
			  impact compared to image with one object, thus not taking mean.
		loss =  conf_loss_scale * conf_loss
				coord_loss_scale * coord_loss 
				class_loss_scale * class_loss

		# TODO: implement for multiple anchors where anchor width and height is defined
	"""
	def __init__(self, reduction='none', class_weight=[1, 5], conf_weight=1):
		super(YoloLoss, self).__init__()
		msg = '[ClassWeights] = {}'.format(class_weight)
		logger.info(msg); print(msg)

		# Confidence loss <BCEWithLogitsLoss> performs sigmoid as well
		self.conf_loss = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.tensor(conf_weight).float())
		self.conf_loss_scale = 0.5
		
		# Class loss where it is multi-label i.e. each label is stand alone.
		self.class_weight = torch.tensor(class_weight).float() # dtype=torch.float  # [1, 5] 1 for first label and 5 for the second lable
		self.class_losses = []
		for i in range(len(self.class_weight)):
			self.class_losses.append(nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.class_weight[i:i+1]))
		self.class_loss_scale = 1 # 

		# Coordinate loss
		self.coord_loss = nn.MSELoss(reduction=reduction)
		self.coord_loss_scale = 5

	def forward(self, output, y_target):
		"""
			Parameters:
			-----------
			output (torch.Tensor): 
			y_target (dict): {'conf': tensor, 'box': tensor, 'label': tensor} (batch x rows x cols x #Anchors x #output_channel)

		"""
		# print('\n\noutput = ', output.shape, 'target = ', len(y_target))
		true_conf = y_target['conf'].float() # [..., 0:1]
		true_box = y_target['box'].float() # [..., 1:5]
		true_label = y_target['label'].float() # [..., 5:y_target.shape[-1]]

		pred_conf = output[..., 0:1]
		pred_box = output[..., 1:5]
		pred_label = output[..., 5:output.shape[-1]]


		# Mask to filter absence of object in a cell
		obj_mask = true_conf.float()

		# Note: Object/no-object loss
		#  print('type(pred_conf) = {}, type(true_conf) = {}'.format(pred_conf.type(), true_conf.type()))
		conf_loss_val = self.conf_loss(pred_conf, true_conf)
		

		# Note: Box and class loss which is included only where there is target
		coord_loss_val = self.coord_loss(pred_box, true_box)
		coord_loss_val = coord_loss_val.sum(dim=-1)
		coord_loss_val = coord_loss_val.unsqueeze(dim=-1)
		
		# Different classes loss where Pr/Ab is checked using sigmoid
		class_loss_val = None


		for i in range(true_label.shape[-1]):
			temp_class_loss_val = self.class_losses[i](pred_label[..., i:i+1], true_label[..., i:i+1])

			if class_loss_val is None:
				class_loss_val = (obj_mask  * temp_class_loss_val).sum() 
			else:
				class_loss_val += (obj_mask  * temp_class_loss_val).sum() 

		loss = (self.conf_loss_scale * conf_loss_val.sum()) \
			 + (self.coord_loss_scale * (obj_mask *  coord_loss_val).sum()) \
			 + (self.class_loss_scale * class_loss_val)
		
		
		if math.isnan(loss.item()):
			print('loss = ', loss.item())
			print('output = ', output)
			print('y_target = ', y_target)
		
		return loss
