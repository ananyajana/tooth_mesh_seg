from visdom import Visdom
import socket
import numpy as np
import os
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def get_avail_gpu():
    '''
    works for linux
    '''
    result = os.popen("nvidia-smi").readlines()

    try:
    # get Processes Line
        for i in range(len(result)):
            if 'Processes' in result[i]:
                process_idx = i

        # get # of gpus
        num_gpu = 0
        for i in range(process_idx+1):
            if 'MiB' in result[i]:
                num_gpu += 1
        gpu_list = list(range(num_gpu))

        # dedect which one is busy
        for i in range(process_idx, len(result)):
            if result[i][22] == 'C':
                gpu_list.remove(int(result[i][5]))
                
        return (gpu_list[0])
    except:
        print('no gpu available, return 0')
        return 0


def setup_logger(save_dir=None, checkpoint=False):
    mode = 'a' if checkpoint else 'w+'
    #print('mode is :', mode)
    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    #fle = Path('{:s}/train_log.txt'.format(save_dir))
    #fle.touch(exist_ok=True)
    
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(save_dir), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)

    #fle2 = Path('{:s}/epoch_results.txt'.format(save_dir))
    #fle2.touch(exist_ok=True)
    
    # set up logger for each result
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(save_dir), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(save_dir))
    if mode == 'w':
        logger_results.info('epoch   Train_loss  train_acc  || Test_loss   test_acc  test_auc')

    return logger, logger_results
