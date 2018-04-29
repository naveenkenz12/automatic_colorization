import sys
from solver import Solver
from utils import process_config


common_params, dataset_params, net_params, solver_params = process_config(conf_file)
if len(str(common_params['gpus']).split(','))==1:
  solver = Solver(True, common_params, solver_params, net_params, dataset_params)
