import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset2 import *
#from meshsegnet import *
#from pointnet2 import *
#from pointnet2_sem import *
#from dgcnn import *
#from gac import *
#from tsgcn import *
from pointMLP import pointMLP_seg
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
import time
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
add_log = True

# set up logger
save_dir='./logs'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

logger, logger_results = utils.setup_logger(save_dir, checkpoint=False)

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="7"
use_visdom = False # if you don't use visdom, please set to False

previous_check_point_path = './models'
previous_check_point_name = 'latest_checkpoint.tar'

#train_list='../pred_58/data/tooth_data_folds_revised_16k_8_class_shuffled_tsgcn/train1.h5'
#val_list='../pred_58/data/tooth_data_folds_revised_16k_8_class_shuffled_tsgcn/val1.h5'
train_list='../pred_58/data/tooth_data_folds_revised_16k_8_class_shuffled_cntrd_bbx_cropped_3dteethseg/train3.h5'
val_list='../pred_58/data/tooth_data_folds_revised_16k_8_class_shuffled_cntrd_bbx_cropped_3dteethseg/val3.h5'

model_path = './models/'
model_name = 'Mesh_Segementation_MeshSegNet_15_classes_60samples' # need to define
checkpoint_name = 'latest_checkpoint.tar'

# DGCNN specific
k = 20
# k = 30 in TSGCNet
emb_dims = 1024
dropout = 0.5
#use_sgd = True
use_sgd = False
lr = 0.001
momentum = 0.9
#scheduler = 'cos'
scheduler = 'step'
seed = 1

#num_classes = 15
num_classes = 8
#num_channels = 15 #number of features
num_channels=24
#num_channels=9
#num_epochs = 200
#num_epochs = 1000
num_epochs = 800
num_workers = 0
#train_batch_size = 10
#val_batch_size = 10
#train_batch_size = 14
#val_batch_size = 14
#train_batch_size = 40
#val_batch_size = 40
train_batch_size = 24
val_batch_size = 24
#train_batch_size = 32
#val_batch_size = 32
#train_batch_size = 16
#val_batch_size = 16
#train_batch_size = 12
#val_batch_size = 12
#num_batches_to_print = 20
num_batches_to_print = 2000
#num_batches_to_print = 1
use_amp = True


if use_visdom:
    # set plotter
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=model_name)

if not os.path.exists(model_path):
    os.mkdir(model_path)

torch.manual_seed(seed)

# set dataset
# we will set the patch size to be bigger because our meshes are ~30k cells
training_dataset = Mesh_Dataset(data_list_path=train_list,
                                num_classes=num_classes,
                                #patch_size=12000)
                                #patch_size=22000)
                                patch_size=16000)
                                #patch_size=2048)
                                #patch_size=200)

val_dataset = Mesh_Dataset(data_list_path=val_list,
                           num_classes=num_classes,
                           #patch_size=12000)
                           #patch_size=22000)
                           #patch_size=2048)
                           patch_size=16000)
                           #patch_size=200)

train_loader = DataLoader(dataset=training_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=num_workers)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=num_workers)

torch.cuda.is_available()

# set model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = PointNet_Seg(num_classes=num_classes, channel=num_channels).to(device, dtype=torch.float)
#model = Pointnet2_seg(num_classes=num_classes, num_channels=num_channels)
#model = GAC_seg(num_classes=num_classes, num_channels=num_channels)
#model = TSGCN_seg(num_classes=num_classes, num_channels=num_channels)
#model = TSGCN_seg2(num_classes=num_classes, num_channels=num_channels)
model = pointMLP_seg(num_classes=num_classes, num_channels=num_channels)
#model = CurveNet(num_classes=num_classes)
#model = model.cuda()
model = nn.DataParallel(model)
model = model.to(device, dtype=torch.float)
#opt = optim.Adam(model.parameters(), amsgrad=True)

opt = None
if use_sgd:
    logger.info("Use SGD")
    opt = optim.SGD(model.parameters(), lr=lr*100, momentum=momentum, weight_decay=1e-4)
else:
    logger.info("Use Adam")
    #opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    opt = optim.Adam(model.parameters(), lr=lr)
    #opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)

scheduler = None
if scheduler == 'cos':
    scheduler = CosineAnnealingLR(opt, epochs, eta_min=1e-3)
elif scheduler == 'step':
    scheduler = StepLR(opt, 120, 0.5, epochs)


scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

losses, mdsc, msen, mppv = [], [], [], []
val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

best_val_dsc = 0.0
#if load_checkpt is True
# re-load
'''
checkpoint = torch.load(os.path.join(previous_check_point_path, previous_check_point_name), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_init = checkpoint['epoch']
losses = checkpoint['losses']
mdsc = checkpoint['mdsc']
msen = checkpoint['msen']
mppv = checkpoint['mppv']
val_losses = checkpoint['val_losses']
val_mdsc = checkpoint['val_mdsc']
val_msen = checkpoint['val_msen']
val_mppv = checkpoint['val_mppv']
del checkpoint

best_val_dsc = max(val_mdsc)
print('previous best value: ', best_val_dsc)
'''
#cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#print('Training model...')
logger.info('Training model...')
#class_weights = torch.ones(15).to(device, dtype=torch.float)
class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
class_weights_for_loss = class_weights
#
#0 label: [223429], 1 label: [56102], 2 label: [61624], 3 label: [39973], 4 label: [39815], 5 label: [34786], 6 label: [28905], 7 label: [27466]

#num_per_class = np.array([223429, 56102, 61624, 39973, 39815, 34786, 28905, 27466])
#weight = num_per_class / float(sum(num_per_class))
#class_weights_for_loss = torch.from_numpy(1 / (weight + 0.02)).to(device, dtype=torch.float)


#num = (1 + 5 + 5 + 7 + 7 + 8 + 15 + 15)
#class_weights_np = np.array((1/36, 5/36, 5/36, 5/36, 5/36, 5/36, 5/36, 5/36))
#class_weights_np = np.array((1/num, 5/num, 5/num, 7/num, 7/num, 8/num, 15/num, 15/num))
#print('class_weights_np: ', class_weights_np)
#class_weights_for_loss = torch.from_numpy(class_weights_np).to(device, dtype=torch.float)
print('class_weights_for_loss: ', class_weights_for_loss)

# batch accumulation parameter
accum_iter = 16

for epoch in range(num_epochs):

    # training
    model.train()
    running_loss = 0.0
    running_mdsc = 0.0
    running_msen = 0.0
    running_mppv = 0.0
    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()

    end = time.time()


    for i_batch, batched_sample in enumerate(train_loader):

        # send mini-batch to device
        inputs = batched_sample['cells'].to(device, dtype=torch.float)
        labels = batched_sample['labels'].to(device, dtype=torch.long)
        #A_S = batched_sample['A_S'].to(device, dtype=torch.float)
        #A_L = batched_sample['A_L'].to(device, dtype=torch.float)
        one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        #outputs = model(inputs, A_S, A_L)
        #with torch.cuda.amp.autocast(enabled=use_amp):
        #print('inputs shape: ', inputs.size())
        #raise ValueError("Exit!")
        outputs = model(inputs)
        #op = outputs.contiguous().view(-1, num_classes)
        op = outputs.view(-1, num_classes)
        lbl = labels.view(-1)

        #loss = F.nll_loss(op, lbl, weight=class_weights)
        loss = F.nll_loss(op, lbl, weight=class_weights_for_loss)
        #print('loss is: ', loss.size())
        #raise ValueError("Exit!")
        #loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
        # normalize loss to account for batch accumulation
        #loss = loss / accum_iter 

        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

        loss.backward()
        opt.step()

        #scaler.scale(loss).backward()
        #scaler.step(opt)
        #scaler.update()
        #opt.zero_grad()
        
        # weights update
        #if ((i_batch + 1) % accum_iter == 0) or (i_batch + 1 == len(train_loader)):
        #    opt.step()
        #    opt.zero_grad()
        # print statistics
        running_loss += loss.item()
        running_mdsc += dsc.item()
        running_msen += sen.item()
        running_mppv += ppv.item()
        loss_epoch += loss.item()
        mdsc_epoch += dsc.item()
        msen_epoch += sen.item()
        mppv_epoch += ppv.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))

        if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
            #print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))
            logger.info('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))
            running_loss = 0.0
            running_mdsc = 0.0
            running_msen = 0.0
            running_mppv = 0.0

    # record losses and metrics
    losses.append(loss_epoch/len(train_loader))
    mdsc.append(mdsc_epoch/len(train_loader))
    msen.append(msen_epoch/len(train_loader))
    mppv.append(mppv_epoch/len(train_loader))
    
    if add_log is True:
        writer.add_scalar("Loss/train", loss_epoch/len(train_loader), epoch)
        writer.add_scalar("mdsc/train", mdsc_epoch/len(train_loader), epoch)
        writer.add_scalar("msen/train", msen_epoch/len(train_loader), epoch)
        writer.add_scalar("mppv/train", mppv_epoch/len(train_loader), epoch)

    if epoch+1 % 20 == 0:
        scheduler.step()
        '''
        if scheduler == 'cos':
            scheduler.step()
        elif scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        '''

    #reset
    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0
    running_mppv = 0.0
    # validation
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        running_val_mdsc = 0.0
        running_val_msen = 0.0
        running_val_mppv = 0.0
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0

        batch_time2 = utils.AverageMeter()
        end = time.time()

        for i_batch, batched_val_sample in enumerate(val_loader):

            # send mini-batch to device
            inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
            labels = batched_val_sample['labels'].to(device, dtype=torch.long)
            #A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
            #A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            #outputs = model(inputs, A_S, A_L)
            #with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            #op = outputs.contiguous().view(-1, num_classes)
            op = outputs.view(-1, num_classes)
            lbl = labels.view(-1)

            #loss = F.nll_loss(op, lbl, weight=class_weights)
            loss = F.nll_loss(op, lbl, weight=class_weights_for_loss)
            #loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

            running_val_loss += loss.item()
            running_val_mdsc += dsc.item()
            running_val_msen += sen.item()
            running_val_mppv += ppv.item()
            val_loss_epoch += loss.item()
            val_mdsc_epoch += dsc.item()
            val_msen_epoch += sen.item()
            val_mppv_epoch += ppv.item()

            # measure elapsed time
            batch_time2.update(time.time() - end)
            end = time.time()


            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                #print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print, batch_time2.avg))
                logger.info('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print, batch_time2.avg))
                running_val_loss = 0.0
                running_val_mdsc = 0.0
                running_val_msen = 0.0
                running_val_mppv = 0.0

        # record losses and metrics
        val_losses.append(val_loss_epoch/len(val_loader))
        val_mdsc.append(val_mdsc_epoch/len(val_loader))
        val_msen.append(val_msen_epoch/len(val_loader))
        val_mppv.append(val_mppv_epoch/len(val_loader))
        
        if add_log is True:
            writer.add_scalar("Loss/val", loss_epoch/len(val_loader), epoch)
            writer.add_scalar("mdsc/val", mdsc_epoch/len(val_loader), epoch)
            writer.add_scalar("msen/val", msen_epoch/len(val_loader), epoch)
            writer.add_scalar("mppv/val", mppv_epoch/len(val_loader), epoch)

        # reset
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0

        # output current status
        #print('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
        logger.info('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
        if use_visdom:
            plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
            plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
            plotter.plot('SEN', 'train', 'SEN', epoch+1, msen[-1])
            plotter.plot('PPV', 'train', 'PPV', epoch+1, mppv[-1])
            plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
            plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
            plotter.plot('SEN', 'val', 'SEN', epoch+1, val_msen[-1])
            plotter.plot('PPV', 'val', 'PPV', epoch+1, val_mppv[-1])

    # save the checkpoint
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'losses': losses,
                'mdsc': mdsc,
                'msen': msen,
                'mppv': mppv,
                'val_losses': val_losses,
                'val_mdsc': val_mdsc,
                'val_msen': val_msen,
                'val_mppv': val_mppv},
                model_path+checkpoint_name+'{}'.format(epoch+1))

    # save the best model
    if best_val_dsc < val_mdsc[-1]:
        best_val_dsc = val_mdsc[-1]
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+'{}_best.tar'.format(model_name))

    # save all losses and metrics data
    pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
    stat = pd.DataFrame(pd_dict)
    stat.to_csv('losses_metrics_vs_epoch.csv')
