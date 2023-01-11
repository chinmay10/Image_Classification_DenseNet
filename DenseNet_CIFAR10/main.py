### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import  training_configs
from ImageUtils import visualize
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str,default=training_configs['mode'], help="train, test or predict")
parser.add_argument("--data_dir",type=str, default=sys.path[0]+training_configs['data_dir'],help="path to the data")
parser.add_argument("--result_dir",type=str,default=sys.path[0]+training_configs['result__dir'], help="path to save the results")

args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(training_configs)

	if args.mode == 'train':
		#Now, training on whole train dataset after training on train, validation
		#Uncomment Validation to run Validation training

		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		# x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
		
		model.train(x_train, y_train)#, x_valid, y_valid)
		model.evaluate(x_train, y_train,'train')
		# model.evaluate(x_valid, y_valid,'valid')
		model.evaluate(x_test, y_test,'test')

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test,'test')

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)

		# visualizing the first testing image to check your image shape
		visualize(x_test[14], 'test.png')

		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test,200)
		np.save(args.result_dir, predictions)
		

### END CODE HERE

