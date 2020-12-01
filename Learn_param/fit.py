"""Learn ODE parameter from expressional profile
for specific cell, get its gene expression matrix, use the method to fit the parameters for ODE at that stage.
"""

import argparse
import os

import numpy as np
import pandas as pd 

import ..Simulator.backwards_euler


parser = argparse.ArgumentParser("Processing the data from blood development")
parser.add_argument("--data_root", type=str, default="../Data/TrainData", help="specify the data root folder")
parser.add_argument("--expr_mat", type=str, default="Blood_development(Normalized).csv", help="specify the file")
parser.add_argument("--cell", type=str, default="CMP", help="target cell name, used in query data from expr_mat")

args = parser.parse_args()



# load the raw expression matrix for cells
expr_mat = pd.read_csv(os.path.join(args.data_root, args.expr_mat), index_col=0)


# for init state 
X = expr_mat[args.cell]

# clustering for 50 genes that is coexpressed


