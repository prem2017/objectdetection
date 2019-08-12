# -*- coding: utf-8 -*-

""" ©Prem Prakash
	General utility module 
"""

import os
import sys

import random

import pickle
import pandas as pd
import numpy as np
import torch

import logging




# For device agnostic 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#******************** Constants
K_SEARCH_LR = SEARCH_LR = False
get_search_lr_flag = lambda: K_SEARCH_LR



#******************** Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
def reset_logger(filename='train_output.log'):
	logger.handlers = []
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def add_logger(filename):
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def setup_logger(filename='output.log'):
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'a'))




#******************** Batches
K_TRAIN_BATCH_SIZE = 16
get_train_batch_size = lambda: K_TRAIN_BATCH_SIZE

K_VALIDATION_BATCH_SIZE = 16
get_val_batch_size = lambda: K_VALIDATION_BATCH_SIZE

K_TEST_BATCH_SIZE = 16
get_test_batch_size = lambda: K_TEST_BATCH_SIZE


#******************** Data
K_PROJECT_DIR = os.path.dirname(os.getcwd())
get_project_dir = lambda: K_PROJECT_DIR
msg = '[Project] dir = {}'.format(K_PROJECT_DIR)
print(msg)

K_DATA_DIR = os.path.join(K_PROJECT_DIR, 'data')
get_root_data_dir = lambda: K_DATA_DIR

get_train_datapath = lambda: os.path.join(K_DATA_DIR, 'train')
get_train_fnames = lambda: [fname for fname in os.listdir(get_train_datapath()) if '.jpg' in fname.lower()]


get_val_datapath = lambda: os.path.join(K_DATA_DIR, 'val')
get_val_fnames = lambda: [fname for fname in os.listdir(get_val_datapath()) if '.jpg' in fname.lower()]

get_test_datapath = lambda: os.path.join(K_DATA_DIR, 'test')
get_test_fnames = lambda: [fname  for fname in os.listdir(get_test_datapath()) if '.jpg' in fname.lower()]

# Pickle file stores all the cordinates and label of the object in that box
get_obj_coord_pickle_datapath = lambda: os.path.join(K_DATA_DIR, 'agri_dict.pk')



#******************** Models and Results
get_models_dir = lambda: os.path.join(K_PROJECT_DIR, 'models')
get_models_dir_old = lambda: os.path.join(K_PROJECT_DIR, 'models') # wt_good_90_1e4_w_1d1_1d5') # wt_deeper_110_1e4_w_1d1_1d5') # plain_deeper_110_1e4') # wt_less_iter_60_1e4_conf_w1d5_class_w1d2_w1d7') # wt_iter_80_conf_w1d4_class_w1d4_w2d8 # second_110_1e5, first_90, first_110, second_90_1e4, deeper_90

get_results_dir = lambda arg='': os.path.join(K_PROJECT_DIR, 'results' , arg)

K_TRAINED_MODELNAME = 'yolo_agri.model'
def set_trained_model_name(rows, cols, anchors=1):
	global K_TRAINED_MODELNAME
	K_TRAINED_MODELNAME = "yolo_trained_on_grid_of_rows_{}_cols_{}_anchors_{}.model".format(rows, cols, anchors)
	return K_TRAINED_MODELNAME

get_trained_model_name = lambda: K_TRAINED_MODELNAME





#******************** Pickle py objects
class PickleHandler(object):    
	@staticmethod
	def dump_in_pickle(py_obj, filepath):
		"""Dumps the python object in pickle
			
			Parameters:
			-----------
				py_obj (object): the python object to be pickled.
				filepath (str): fullpath where object will be saved.
			
			Returns:
			--------
				None
		"""
		with open(filepath, 'wb') as pfile:
			pickle.dump(py_obj, pfile)
	
	
	
	@staticmethod
	def extract_from_pickle(filepath):
		"""Extracts python object from pickle
			
			Parameters:
			-----------
				filepath (str): fullpath where object is pickled
			
			Returns:
			--------
				py_obj (object): python object extracted from pickle
		"""
		with open(filepath, 'rb') as pfile:
			py_obj = pickle.load(pfile)
			return py_obj    


#******************** Better presentation of printed dict
import collections
def pretty(d, indent=0):
	""" Pretty printing of dictionary """
	ret_str = ''
	for key, value in d.items():

		if isinstance(value, collections.Mapping):
			ret_str = ret_str + '\n' + '\t' * indent + str(key) + '\n'
			ret_str = ret_str + pretty(value, indent + 1)
		else:
			ret_str = ret_str + '\n' + '\t' * indent + str(key) + '\t' * (indent + 1) + ' => ' + str(value) + '\n'

	return ret_str



# [How to use a dot “.” to access members of dictionary?] 
# (https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary)
class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


class DotDict(dict): 
	"""dot.notation access to dictionary attributes""" 
	def __getattr__(*args): 
		val = dict.get(*args) 
		return DotDict(val) if type(val) is dict else val
	
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__



#******************** Test utitlity
if __name__ == '__main__':
	print('Util is the main module')

	nested_dict = {'val':'nested works too'}
	mydict = dotdict(nested_dict)
	print(mydict.val)


	d = {'foo': {'bar': 'baz'}}
	d = DotDict(d)
	print(d.foo.bar)

