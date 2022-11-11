#from open3d import linux as open3d
from os.path import join
import numpy as np
#import colorsys, random, os, sys
import pandas as pd

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#sys.path.append(BASE_DIR)
#sys.path.append(os.path.join(BASE_DIR, 'utils'))

#import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
#import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

class ConfigTooth:
    #k_n = 4  # KNN
    #k_n = 24  # KNN
    k_n = 32 # KNN
    #k_n = 8  # KNN
    #k_n = 16  # KNN
    #num_layers = 4  # Number of layers
    #num_layers = 2  # Number of layers
    num_layers = 3  # Number of layers
    #num_layers = 1  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 8  # Number of valid classes
    #num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 750  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    #d_out = [16, 16, 16, 16]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8
