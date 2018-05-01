import sys
from solver import Solver
#from utils import process_config

params = {}
#common_params, dataset_params, net_params, solver_params = process_config(conf_file)
params["batch_size"] = 10
params['image_size'] = 224
params['learning_rate'] = 1
params['moment'] = 1
params['max_iterators'] = 10
params['train_dir'] = '../datasets/'
params['path'] = "../datasets/"
params['lr_decay'] = 0.5
params['decay_steps'] = 10

#if len(str(common_params['gpus']).split(','))==1:
solver = Solver(True, params)#, solver_params, net_params, dataset_params)
solver.train_model()