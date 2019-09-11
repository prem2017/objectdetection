# -*- coding: utf-8 -*-


""" ©Prem Prakash
	Trainer: trains the network and saves the model for different check points
"""


import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as torch_transformer
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchcontrib.optim import SWA


from .loss_functions import YoloLoss
from .models import AgriNet
from .image_dataset import ImageDataset

from .predictor import gen_conf_and_cls_report

from . import util
from .util import logger
from . import yolo_utils as yutil


class Optimizer(object):
	""" Different optimizer of optimize learning process than vanilla greadient descent """
	def __init__(self):
		super(Optimizer, self).__init__()
		
		
	@staticmethod
	def rmsprop_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.RMSprop(params=params, lr=lr, alpha=0.99, eps=1e-6, centered=True, weight_decay=weight_decay)


	@staticmethod
	def adam_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)

	@staticmethod
	def sgd_optimizer(params, lr=1e-6, weight_decay=1e-6, momentum=0.9):
		return optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=momentum)




def train_network(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs=90, sanity_check=False):
	"""Trains the network and saves for different checkpoints such as minimum train/val loss, f1-score, different metric value

		Parameters:
		-----------
			dataloader (dict): {key (str):  value(torch.utils.data.DataLoader)} training and validation dataloader to respective purposes
			model (dict): {key (str): value (nn.Module)} different models if need to train with architecture like encoder/decoder
			loss_function (torch.nn.Module): module to mesure loss between target and model-output
			optimizer (dict): {key (str): value (Optimizer)} non vanilla gradient descent method to optimize learning and descent direction
			start_lr (float): For one cycle training the start learning rate
			end_lr (float): the end learning must be greater than start learning rate
			num_epochs (int): number of epochs the one cycle is 
			sanity_check (bool): if the training is perfomed to check the sanity of the model. i.e. to anaswer 'is model is able to overfit for small amount of data?'

		Returns:
		--------
			None: perfoms the required task of training


		TODO: better manage the one cycle learning setter and getter. 

	"""


	model = {k: v.train() for k, v in model.items()}
	logger_msg = '\nDataLoader = %s' \
				 '\nModel = %s' \
				 '\nLossFucntion = %s' \
				 '\nOptimizer = %s' \
				 '\nStartLR = %s, EndLR = %s' \
				 '\nNumEpochs = %s' % (dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs)

	logger.info(logger_msg), print(logger_msg)

	# [https://arxiv.org/abs/1803.09820]
	# This is used to find optimal learning-rate which can be used in one-cycle training policy
	# [LR]TODO: for finding optimal learning rate
	if util.get_search_lr_flag():
		lr_scheduler = [MultiStepLR(optimizer=opt, milestones=list(np.arange(2, 24, 2)), gamma=10, last_epoch=-1)
						for k, opt in optimizer.items()]
		

	# [New] 
	# https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
	swa_list = [SWA(base_opt) for _, base_opt in optimizer.items()] # use in manual model

	def get_lr():
		lr = []
		for _, opt in optimizer.items():
			for param_group in opt.param_groups:
				lr.append(np.round(param_group['lr'], 11))
		return lr

	def set_lr(lr):
		for _, opt in optimizer.items():
			for param_group in opt.param_groups:
				param_group['lr'] = lr

	# Loss storation
	current_epoch_batchwise_loss = []
	avg_epoch_loss_container = []  # Stores loss for each epoch averged over
	all_epoch_batchwise_loss = []
	avg_val_loss_container = []
	val_report_container = []
	f1_checker_container = []
	aps_container = []


	if util.get_search_lr_flag():
		extra_epochs = 4
	else:
		extra_epochs = 20
	total_epochs = num_epochs + extra_epochs

	# One cycle setting of Learning Rate
	num_steps_upndown = 10
	further_lowering_factor = 10
	further_lowering_factor_steps = 4

	def one_cycle_lr_setter(current_epoch):
		if current_epoch <= num_epochs:
			assert end_lr > start_lr, '[EndLR] should be greater than [StartLR]'
			lr_inc_rate = np.round((end_lr - start_lr) / (num_steps_upndown), 9)
			lr_inc_epoch_step_len = max(num_epochs / (2 * num_steps_upndown), 1)

			steps_completed = current_epoch / lr_inc_epoch_step_len
			print('[Steps Completed] = ', steps_completed)
			if steps_completed <= num_steps_upndown:
				current_lr = start_lr + (steps_completed * lr_inc_rate)
			else:
				current_lr = end_lr - ((steps_completed - num_steps_upndown) * lr_inc_rate)
			set_lr(current_lr)
		else:
			current_lr = start_lr / (
						further_lowering_factor ** ((current_epoch - num_epochs) // further_lowering_factor_steps))
			set_lr(current_lr)

	if sanity_check:
		train_dataloader = next(iter(dataloader['train']))
		train_dataloader = [train_dataloader] * 32
	else:
		train_dataloader = dataloader['train']

	for epoch in range(total_epochs):
		msg = '\n\n\n[Epoch] = %s' % (epoch + 1)
		print(msg)
		start_time = time.time()
		start_datetime = datetime.now()
		
		for i, (x, y, _) in enumerate(train_dataloader): # _: image size
			loss = 0


			x = x.to(device=util.device, dtype=torch.float)
			for k, v in y.items():
				y[k] = v.to(device=util.device)
			
			# TODO: early breaker
			# if i == 2:
			# 	print('[Break] by force for validation check')
			# 	break

			for _, opt in optimizer.items():
				opt.zero_grad()
			
			output = model['yolo'](x) # initial_states[-1, :, :] i.e shape => [-1 (#layers) x batch-size x hidden-size]


			loss = loss_function(output, y)

			loss.backward()

			# TODO: Either use Optimizer step or SWA not both
			# for _, opt in optimizer.items():
			# 	opt.step()

			# [New] Optimizer step
			for opt in swa_list:
				opt.step()

			if i > 20 and i % 5 == 0:
				for opt in swa_list:
					opt.update_swa()


			current_epoch_batchwise_loss.append(loss.item())
			all_epoch_batchwise_loss.append(loss.item())

			batch_run_msg = '\nEpoch: [%s/%s], Step: [%s/%s], InitialLR: %s, CurrentLR: %s, Loss: %s' \
							% (epoch + 1, total_epochs, i + 1, len(train_dataloader), start_lr, get_lr(), loss.item())
			print(batch_run_msg)
		#------------------ End of an Epoch ------------------ 
		
		# store average loss
		avg_epoch_loss = np.round(sum(current_epoch_batchwise_loss) / (i + 1.0), 6)
		current_epoch_batchwise_loss = []
		avg_epoch_loss_container.append(avg_epoch_loss)
		
		if not (util.get_search_lr_flag() or sanity_check):
			val_loss, val_report, f1_checker, aps = calc_validation_loss(model, dataloader['val'], loss_function)

		if not (util.get_search_lr_flag() or sanity_check):
			avg_val_loss_container.append(val_loss)
			val_report_container.append(val_report)  # ['epoch_' + str(epoch)] = val_report
			f1_checker_container.append(f1_checker)	
			aps_container.append(aps)

			if np.round(val_loss, 4) <= np.round(min(avg_val_loss_container), 4):
				model = save_model(model, extra_extension='_minval') # + '_epoch_' + str(epoch))

			if np.round(aps, 4) >= np.round(max(aps_container), 4):
				model = save_model(model, extra_extension='_maxaps') # + '_epoch_' + str(epoch))

			if np.round(f1_checker[0], 4) >= np.round(np.array(f1_checker_container)[:, 0].max(), 4):
				model = save_model(model, extra_extension='_maxconf_f1') # + '_epoch_' + str(epoch))


		if avg_epoch_loss <= min(avg_epoch_loss_container):
			model = save_model(model, extra_extension='_mintrain')


		
		# Logger msg
		msg = '\n\n\n\n\nEpoch: [%s/%s], InitialLR: %s, CurrentLR= %s \n' \
			  '\n\n[Train] Average Epoch-wise Loss = %s \n' \
			  '\n\n[Validation] Average Epoch-wise loss = %s \n' \
			  '\n\n[Validation] Report = %s \n'\
			  '\n\n[Validation] F-Report = %s\n'\
			  %(epoch+1, total_epochs, start_lr, get_lr(), avg_epoch_loss_container, avg_val_loss_container, None if not val_report_container else util.pretty(val_report_container[-1]), f1_checker_container)
		logger.info(msg); print(msg)
		
		if avg_epoch_loss < 1e-6 or get_lr()[0] < 1e-11 or get_lr()[0] >= 10:
			msg = '\n\nAvg. Loss = {} or Current LR = {} thus stopping training'.format(avg_epoch_loss, get_lr())
			logger.info(msg)
			print(msg)
			break
			
		
		# [LR]TODO:
		if util.get_search_lr_flag():
			for lr_s in lr_scheduler:
				lr_s.step(epoch+1) # TODO: Only for estimating good learning rate
		else:
			one_cycle_lr_setter(epoch + 1)

		end_time = time.time()
		end_datetime = datetime.now()
		msg = '\n\n[Time] taken for epoch({}) time = {}, datetime = {} \n\n'.format(epoch+1, end_time - start_time, end_datetime - start_datetime)
		logger.info(msg); print(msg)

	# ----------------- End of training process -----------------

	msg = '\n\n[Epoch Loss] = {}'.format(avg_epoch_loss_container)
	logger.info(msg)
	print(msg)

	
	# [LR]TODO: change for lr finder
	if util.get_search_lr_flag():
		losses = avg_epoch_loss_container
		plot_file_name = 'training_epoch_loss_for_lr_finder.png'
		title = 'Training Epoch Loss'
	else:
		losses = {'train': avg_epoch_loss_container, 'val': avg_val_loss_container}
		plot_file_name = 'training_vs_val_avg_epoch_loss.png'
		title= 'Training vs Validation Epoch Loss'
	plot_loss(losses=losses,
			plot_file_name=plot_file_name,
			title=title)
	plot_loss(losses=all_epoch_batchwise_loss, plot_file_name='training_batchwise.png', title='Training Batchwise Loss',
			xlabel='#Batchwise')
			
	
	
	# Save the model
	# [New] for now can only save in the end
	# for _, model_local in model.items():
	# 	print('[Before]')
	# 	for param in model_local.parameters():
	#   		print(param.data)

	# Swap the weights # TODO: may be not use stochastic summation of weights
	for opt in swa_list:
		opt.swap_swa_sgd()
	model = save_model(model)

	# for _, model_local in model.items():
	# 	print('[After]')
	# 	for param in model_local.parameters():
	# 		print(param.data)
	



def calc_validation_loss(model, val_dataloader, loss_func):
	"""Computes validation loss on developement data also called validation data

		Parameters:
		-----------
			model (dict): {key (str): value (nn.Module)} different models if need to train with architecture like encoder/decoder
			val_dataloader (torch.utils.data.DataLoader): validation dataloader to check how is training performing
			loss_function (torch.nn.Module): module to mesure loss between target and model-output

		Returns:
		--------
			val_loss (float): validation loss like training loss
			val_report (dict): {key (str): value (str)} some of reports are like confusion-matrix, 
							   classification-report, f1-score, average-precsion-score (aps) for each of the classes 
			f1_checker (list[list]): A seperately stored f1-values for different classes and confidence  
			aps (float): Average of aps on all of the classes 
	"""
	
	model = {k: v.eval() for k, v in model.items()}
	
	loss_val = 0
	f1_score_conf = 0
	f1_score_object = 0

	y_pred_all = None
	y_true_all = None
	with torch.no_grad():
		for i, (x, y, _) in enumerate(val_dataloader): # last one is true image size
			print('[val] i = ', i)
			loss_val = 0
			
			x = x.to(device=util.device, dtype=torch.float)
			for k, v in y.items():
				y[k] = v.to(device=util.device)
			
			y_pred = model['yolo'](x)
			loss = loss_func(y_pred, y)

			loss_val += loss.item()

			y_pred_all = y_pred if y_pred_all is None else torch.cat((y_pred_all, y_pred), dim=0)
			if y_true_all is None:
				y_true_all = {}
				for k, v in y.items():
					y_true_all[k] = v
			else:
				for k, v in y.items():
					y_true_all[k] = torch.cat((y_true_all[k], v), dim=0)

	val_report, f1_checker, aps = gen_conf_and_cls_report(y_true_all, y_pred_all)
	model = {k: v.train() for k, v in model.items()}
	val_loss =  np.round(loss_val / (i + 1.0), 6)
	return val_loss, val_report, f1_checker, aps  # val_loss, val_score, model






# Plot training loss
def plot_loss(losses, plot_file_name='training_loss.png', title='Training Loss', xlabel='Epochs'):
	"""Plots the loss fo the given type and saves them

		Parameters:
		-----------
			losses (dict/list): if dict then it normally plotting training training and validation in same plot
			plot_file_name (str): filename with which plot will be saved
			title (str): title of the plot
			xlabel (str): x-label indicating if it is per epoch or per batchwise

		Returns:
		--------
			None: 

	"""
	fig = plt.figure()
	label_key = {'train': 'Training Loss', 'val': 'Validation Loss'}
	if isinstance(losses, dict):
		for k, v in losses.items():
			plt.plot(range(1, len(v)), v[1:], '-*', markersize=3, lw=1, alpha=0.6, label=label_key[k])	
	else:
		plt.plot(range(1, len(losses)+1), losses, '-*', markersize=3, lw=1, alpha=0.6)
	
	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Mean Square Loss (MSE)')
	plt.legend(loc='upper right')
	full_path = os.path.join(util.get_results_dir(), plot_file_name)
	fig.tight_layout()  # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
	fig.savefig(full_path)
	plt.close(fig)  # clo


def save_model(models_dict, extra_extension=""):
	"""Saves the model by first bringing it on CPU and reverting 

		Parameters:
		-----------
			models_dict (dict): {key (str): value (torch.nn.Module)} saves the model at model directory
			extra_extension (str): gives extra details about when and what triggered the save of the model like min trainloss/valloss, max f1score/aps etc.

		Returns:
		--------
			models_dict (dict): {key (str): value (torch.nn.Module)} reverts the model back on the given device, basically resotres the state of the model for returning 	

	"""

	msg = '[Save] model extra_extension = {}'.format(extra_extension)
	logger.info(msg); print(msg)

	model_path = os.path.join(util.get_models_dir(), util.get_trained_model_name()) + extra_extension
	msg = '[SaveModel] saved model name = {}'.format(model_path)
	logger.info(msg); print(msg)


	save_dict = {}
	for k, model in models_dict.items():
		if next(model.parameters()).is_cuda:
			model = model.cpu().float()
		save_dict[k] = model.state_dict()
		
	torch.save(save_dict, model_path)
	
	models_dict = {k: v.to(util.device) for k, v in models_dict.items()}
	return models_dict



# Pre-requisite setup for training process
def train_model(train_data_info, val_data_info, sanity_check=False):
	"""Setup all the pre-requisites for complete training of the model 

		Parameters:
		-----------
			train_data_info (dict): information needed to setup train dataset such datapath etc.
			val_data_info (dict): information needed to setup val dataset such datapath etc.
			sanity_check (bool): pass the boolean to the method <train_network> to indicate if it is sanity check or full training

		Returns:
		--------
			None: Only works as setup for the training of the model

		TODO: either also bring the <lr-finder-flag> here or put the <sanity_check> float also in util or make some setup file
	"""
	msg = '\n\n[Train] data info = {}\n\n[Validation] data info = {}\n\n[SanityCheck] = {}'.format(train_data_info, val_data_info, sanity_check)
	logger.info(msg), print(msg)
	
	train_params = {}
	# [LR]
	if util.get_search_lr_flag() :
		start_lr, end_lr, epochs = 1e-6, 10, 20 # [15230.359358, 15201.38758, 15128.980662, 15057.720209, 14643.659191, 13840.406928, 13773.572943, 13639.902689,  0.1 =>   13874.852725, 13785.437565, 14383.995133, 14044.938432, 109496518.60653]
	else:
		start_lr, end_lr, epochs = 2e-3, 9e-3, 70 #  3e-3, 6e-3, 70  2e-3, 9e-3, 70 # 1e-3, 5e-3, 50 # 7e-3, 11e-3, 70
	train_params['start_lr'] = start_lr = start_lr
	train_params['end_lr'] = end_lr
	train_params['num_epochs'] = epochs


	use_batchnorm = True # TODO: batchnorm
	if sanity_check or util.get_search_lr_flag():
		weight_decay = 0
		dropout = 0
	else:
		weight_decay = 1e-3 # 1e-6
		# dropout = 0.3 # 0.5 # might not be needed
	class_weight = [1.1, 1.4] #  [1.1, 1.5] # [1, 5] TODO: check for other values also 
	conf_weight = 1.1

	#####################
	## TODO: For data augmentation such as flip, hue, saturation, brighness
	#####################
	# transformer = torch_transformer.Compose([RescaleImage((int(1.2 * util.HEIGHT_DEFAULT_VAL), int(1.2 * util.WIDTH_DEFAULT_VAL))),
	# 										 RandomCropImage((util.HEIGHT_DEFAULT_VAL, util.WIDTH_DEFAULT_VAL)),

	# ])
	# transformer = []


	dataset = {}
	train_data_info['model_img_size'] = yutil.get_model_img_size()
	train_data_info['boxes_info_dict'] = util.PickleHandler.extract_from_pickle(util.get_obj_coord_pickle_datapath())
	dataset['train'] = ImageDataset(**train_data_info)
	
	val_data_info['model_img_size'] = yutil.get_model_img_size()
	val_data_info['boxes_info_dict'] = util.PickleHandler.extract_from_pickle(util.get_obj_coord_pickle_datapath())
	dataset['val'] = ImageDataset(**val_data_info)

	dataloader = {}
	dataloader['train'] = DataLoader(dataset=dataset['train'], batch_size=util.get_train_batch_size(), shuffle=True)
	dataloader['val'] = DataLoader(dataset=dataset['val'], batch_size=util.get_val_batch_size())
	train_params['dataloader'] = dataloader

	net_args = {}
	net_args['in_channels'] = 3
	net_args['num_classes'] = yutil.get_num_classes()
	net_args['anchors'] = yutil.get_num_anchors()
	model = {'yolo': AgriNet(**net_args)}
	train_params['model'] = model = {k: v.to(util.device) for k, v in model.items()}

	loss_function = YoloLoss(class_weight=class_weight, conf_weight=conf_weight) 
	train_params['loss_function'] = loss_function.to(util.device)
	

	optimizer = {'yolo': Optimizer.adam_optimizer(params=model['yolo'].parameters(), lr=start_lr, weight_decay=weight_decay)}

	

	train_params['optimizer'] = optimizer
	train_params['sanity_check'] = sanity_check
	# optim.SGD(params=model.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=0.9)
	


	# Train the network
	train_network(**train_params)




if __name__ == '__main__':
	print('Trainer')
	torch.manual_seed(666)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(666)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	util.reset_logger()
	print('Pid = ', os.getpid())


	# Sanity check of the model for learning
	sanity_check = False
	
	# set model name
	rows, cols = yutil.get_grid_shape()
	anchors = yutil.get_num_anchors()
	util.set_trained_model_name(rows, cols, anchors)


	train_data_info = {'datatype_dir': util.get_train_datapath(), 'get_fnames_method': util.get_train_fnames}
	val_data_info = {'datatype_dir': util.get_val_datapath(), 'get_fnames_method': util.get_val_fnames}


	msg = '[Datapath] \nTrain = {}, \nValidation = {}'.format(train_data_info, val_data_info)
	logger.info(msg); print(msg)
	train_model(train_data_info=train_data_info, val_data_info=val_data_info, sanity_check=sanity_check)
	
	msg = '\n\n********************** Training Complete **********************\n\n'
	logger.info(msg); print(msg)





