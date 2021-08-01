# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
import pdb
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_crnn
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
from models.CRNN import CRNN_fpn, CRNN, Predictor, Frame_Discriminator, Clip_Discriminator

import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from torch.autograd import Variable
import random
import torch.nn.functional as F
import collections

def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1, optimizer_d=None, optimizer_crnn=None, c_epoch=None, rampup_value_adv=None):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:
    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # lr_adv = rampup_value_adv * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    # if c_epoch % 50 == 0:
    # lr = lr * 0.1
    # lr_adv = lr_adv * 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay

    if optimizer_d != None:
        for param_group in optimizer_d.param_groups:
            param_group['lr'] = lr * 0.1

    if optimizer_crnn != None:
        for param_group in optimizer_crnn.param_groups:
            param_group['lr'] = lr * 0.1


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("gradient_flow.png")

### ICT
def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch, consistency_rampup_starts, consistency_rampup_ends):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * ramps.sigmoid_rampup(epoch, consistency_rampup_ends - consistency_rampup_starts )

def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, z, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y, z = x.data.cpu().numpy(), y.data.cpu().numpy(), z.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])
    mixed_z = torch.Tensor(lam * z + (1 - lam) * z[index,:])

    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    mixed_z = Variable(mixed_z.cuda())
    return mixed_x, mixed_y, mixed_z, lam

def train(train_loader, model, optimizer, c_epoch, ema_model=None, ema_predictor=None, mask_weak=None, mask_strong=None, adjust_lr=False, discriminator=None, optimizer_d=None, predictor=None, optimizer_crnn=None, ISP=False):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    domain_acc = []

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        if ISP:
            # Generate input random shift feature 
            pooling_time_ratio = 4
            shift_list = [random.randint(-32,32)*pooling_time_ratio for iter in range(cfg.batch_size)]
            freq_shift_list = [random.randint(-4,4) for iter in range(cfg.batch_size)]
            for k in range(cfg.batch_size):
                input_shift = torch.roll(batch_input[k], shift_list[k], dims=1)
                input_shift = torch.unsqueeze(input_shift, 0)
                input_freq_shift = torch.roll(batch_input[k], freq_shift_list[k], dims=2)
                input_freq_shift = torch.unsqueeze(input_freq_shift, 0)
                if k==0:
                    batch_input_shift = input_shift
                    batch_input_freq_shift = input_freq_shift
                else:
                    batch_input_shift = torch.cat((batch_input_shift,input_shift), 0)
                    batch_input_freq_shift = torch.cat((batch_input_freq_shift,input_freq_shift), 0)
                
            batch_input_shift = to_cuda_if_available(batch_input_shift)
            batch_input_freq_shift = to_cuda_if_available(batch_input_freq_shift)
 
        global_step = c_epoch * len(train_loader) + i
        niter = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))

        adv_step = (c_epoch-start_epoch) * len(train_loader) + i
        rampup_value_adv = ramps.exp_rampup(adv_step, cfg.n_epoch_rampup*len(train_loader))

        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value, optimizer_d=optimizer_d, optimizer_crnn=optimizer_crnn, c_epoch=c_epoch, rampup_value_adv=rampup_value_adv)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        # Generate DA labels for training discriminator
        if f_args.level == 'frame':
            domain_label = torch.zeros((24, 157, 2))
            domain_label[:18, :, 1] = 1 # target: 1 for axis 1
            domain_label[18:, :, 0] = 1 # source: 1 for axis 0
        elif f_args.level == 'clip':
            domain_label = torch.zeros((24, 2))
            domain_label[:18, 1] = 1 # target: 1 for axis 1
            domain_label[18:, 0] = 1 # source: 1 for axis 0


        batch_input, ema_batch_input, target, domain_label = to_cuda_if_available(batch_input, ema_batch_input, target, domain_label)
        # Outputs
        if ema_model != None:
            encoded_x_ema, _ = ema_model(ema_batch_input)
            strong_pred_ema, weak_pred_ema = ema_predictor(encoded_x_ema)
            strong_pred_ema = strong_pred_ema.detach()
            weak_pred_ema = weak_pred_ema.detach()
        
        
        adv_w = 1 # weight of adversarial loss
        update_step = 1
        # Update discriminator
        if discriminator is not None:
            if global_step % update_step == 0:
                optimizer_d.zero_grad()
                encoded_x, d_input = model(batch_input)
                domain_pred = discriminator(d_input.detach())

                # To balance source and target amount
                random_choice = np.random.choice(18,6,replace=False)
                choice = np.append(random_choice,[18,19,20,21,22,23])
                domain_pred = domain_pred[choice]
                domain_label_original = domain_label[choice]

                domain_loss_d = adv_w * class_criterion(domain_pred, domain_label_original)
                domain_loss_d.backward()
                optimizer_d.step()
                

        # Update feature extractor
        if discriminator is not None:
            if global_step % update_step == 0:
                optimizer_d.zero_grad()     
                optimizer_crnn.zero_grad()
                encoded_x, d_input = model(batch_input)
                domain_pred = discriminator(d_input)

                # Generate domain labels for training feature extractor
                if f_args.level == 'frame':
                    domain_label = torch.zeros((18, 157, 2))
                    domain_label[:18, :, 0] = 1 # target: 1 for axis 0
                elif f_args.level == 'clip':
                    domain_label = torch.zeros((18, 2))
                    domain_label[:18, 0] = 1
                
                domain_label = to_cuda_if_available(domain_label)
                
                # To balance source and target amount
                choice = np.random.choice(18,12,replace=False)
                domain_pred = domain_pred[choice]
                domain_label_original = domain_label[choice]    


                domain_loss = adv_w * class_criterion(domain_pred, domain_label_original)
                domain_loss.backward()
                optimizer_crnn.step()
        

        # Update feature extractor and predictor
        optimizer.zero_grad()
        if discriminator is not None:
            optimizer_d.zero_grad() 
        optimizer_crnn.zero_grad()
        encoded_x, d_input = model(batch_input)
        strong_pred, weak_pred = predictor(encoded_x)

        if ISP:
            # Prediction and target(strong) shift
            for k in range(cfg.batch_size):
                pool_shift = int(shift_list[k]/pooling_time_ratio)
                pred_shift = torch.roll(strong_pred[k], pool_shift, dims=0)
                pred_shift = torch.unsqueeze(pred_shift, 0)
                target_shift = torch.roll(target[k], pool_shift, dims=0)
                target_shift = torch.unsqueeze(target_shift, 0)
                if k==0:
                    strong_pred_shift = pred_shift
                    strong_target_shift = target_shift
                else:
                    strong_pred_shift = torch.cat((strong_pred_shift,pred_shift), 0)
                    strong_target_shift = torch.cat((strong_target_shift,target_shift), 0)
            strong_pred_shift = strong_pred_shift.detach() 
            # Shifted prediction
            encoded_x_shift, _ = model(batch_input_shift)
            strong_shift_pred, weak_shift_pred = predictor(encoded_x_shift)
            encoded_x_freq_shift, _ = model(batch_input_freq_shift)
            strong_freq_shift_pred, weak_freq_shift_pred = predictor(encoded_x_freq_shift)

            # Setting for ICT
            mask_unlabel = slice(6,18)
            mixup_sup_alpha = 1.0
            mixup_usup_alpha = 2.0
            mixup_consistency = 1.0
            consistency_rampup_starts = 0.0
            consistency_rampup_ends = 100.0 

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            if ISP:
                # Contain weak pseudo-label
                weak_class_loss = class_criterion(torch.cat((weak_pred[mask_weak], weak_pred[mask_unlabel]), 0), torch.cat((target_weak[mask_weak], target_weak[mask_unlabel]), 0))
                
                # SCT
                weak_freq_shift_class_loss = class_criterion(weak_freq_shift_pred[mask_weak], target_weak[mask_weak])
                
                # ICT
                if mixup_sup_alpha:
                    mixed_input_weak, target_a_weak, target_b_weak, lam_weak = mixup_data_sup(batch_input[mask_weak], target_weak[mask_weak], mixup_sup_alpha)
                    encoded_x_weak, _ = model(mixed_input_weak)
                    _, output_mixed_weak = predictor(encoded_x_weak)
                    loss_func_weak = mixup_criterion(target_a_weak, target_b_weak, lam_weak)
                    mixup_weak_class_loss = loss_func_weak(class_criterion, output_mixed_weak)
                    meters.update('maxup_weak_class_loss', mixup_weak_class_loss.item())
            else:
                weak_class_loss = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_weak[mask_weak]} \n "
                          f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")
            meters.update('weak_class_loss', weak_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(strong_pred[mask_strong], target[mask_strong])
            meters.update('Strong loss', strong_class_loss.item())

            if ISP:
                # SCT
                strong_shift_class_loss = class_criterion(strong_shift_pred[mask_strong], strong_target_shift[mask_strong])
                strong_freq_shift_class_loss = class_criterion(strong_freq_shift_pred[mask_strong], target[mask_strong])

                # ICT
                if mixup_sup_alpha:
                    mixed_input_strong, target_a_strong, target_b_strong, lam_strong = mixup_data_sup(batch_input[mask_strong], target[mask_strong], mixup_sup_alpha)
                    encoded_x_strong, _ = model(mixed_input_strong)
                    output_mixed_strong, _ = predictor(encoded_x_strong)
                    loss_func_strong = mixup_criterion(target_a_strong, target_b_strong, lam_strong)
                    mixup_strong_class_loss = loss_func_strong(class_criterion, output_mixed_strong)
                    meters.update('mixup_strong_class_loss', mixup_strong_class_loss.item())        

        # Teacher-student consistency cost
        if ema_model is not None:
            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred, strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())

            if ISP:
                # ICT
                batch_input_u = batch_input[mask_unlabel]
                encoded_x_u, _ = ema_model(batch_input_u)
                ema_logit_unlabeled, ema_logit_unlabeled_weak = ema_predictor(encoded_x_u)
                ema_logit_unlabeled = ema_logit_unlabeled.detach()
                ema_logit_unlabeled_weak = ema_logit_unlabeled_weak.detach()
                
                if mixup_consistency:
                    mixedup_x, mixedup_target, mixedup_target_weak, lam = mixup_data(batch_input_u, ema_logit_unlabeled, ema_logit_unlabeled_weak, mixup_usup_alpha)
                    encoded_x_mixedup, _ = model(mixedup_x)
                    output_mixed_u, output_mixed_u_weak = predictor(encoded_x_mixedup)

                    mixup_consistency_weak_loss = consistency_criterion(output_mixed_u_weak, mixedup_target_weak)# / 12
                    mixup_consistency_strong_loss = consistency_criterion(output_mixed_u, mixedup_target)# / 12
                    meters.update('mixup_cons_weak_loss', mixup_consistency_weak_loss.item())
                    meters.update('mixup_cons_strong_loss', mixup_consistency_strong_loss.item())
                    
                    mixup_consistency_weak_loss = consistency_cost*mixup_consistency_weak_loss     
                    mixup_consistency_strong_loss = consistency_cost*mixup_consistency_strong_loss        
 
 
        # Calculate loss for labeled data
        loss = strong_class_loss + weak_class_loss
        
        if ema_model is not None:
            loss = loss + (consistency_loss_weak + consistency_loss_strong)
        
        if ISP:
            # Add shift consistency loss
            consistency_loss_shift = consistency_cost/2 * consistency_criterion(strong_shift_pred, strong_pred_shift)            
            loss = loss + (weak_freq_shift_class_loss + mixup_weak_class_loss + strong_shift_class_loss + strong_freq_shift_class_loss + mixup_strong_class_loss + mixup_consistency_weak_loss + mixup_consistency_strong_loss + consistency_loss_shift)
        
        
        writer.add_scalar('Loss', loss.item(), niter)
        if discriminator is not None:
            writer.add_scalar('Weak class loss', weak_class_loss.item(), niter)
            
            if global_step % update_step == 0:
                writer.add_scalar('Decoder domain loss', domain_loss_d.item(), niter)   
                writer.add_scalar('Feature extractor domain loss', domain_loss.item(), niter)
            
        writer.add_scalar('Strong class loss', strong_class_loss.item(), niter)

        if ema_model is not None:
            writer.add_scalar('Consistency strong', consistency_loss_strong.item(), niter)
            writer.add_scalar('Consistency weak', consistency_loss_weak.item(), niter)

        if ISP:
            writer.add_scalar('Mixup weak class loss', mixup_weak_class_loss.item(), niter)
            writer.add_scalar('Mixup strong class loss', mixup_strong_class_loss.item(), niter)
            writer.add_scalar('Mixup consistency weak loss', mixup_consistency_weak_loss.item(), niter)
            writer.add_scalar('Mixup consistency strong loss', mixup_consistency_strong_loss.item(), niter)    
            writer.add_scalar('Consistency shift', consistency_loss_shift.item(), niter)
            writer.add_scalar('Strong shift class loss', strong_shift_class_loss.item(), niter)
            writer.add_scalar('Weak freq shift class loss', weak_freq_shift_class_loss.item(), niter)
            writer.add_scalar('Strong freq shift class loss', strong_freq_shift_class_loss.item(), niter)

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        
        # Compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)
            update_ema_variables(predictor, ema_predictor, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss


def get_dfs(desed_dataset, nb_files=None, separated_sources=False):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                       nb_files=nb_files, download=False)
    log.debug(f"synthetic: {synthetic_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                # "eval2018": eval_2018_df
                }

    return data_dfs


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    # pretrain or adaptation
    parser.add_argument("-stage", '--stage', type=str, default='pretrain',
                    help="Choose the training stage of ADDA. 'pretrain' or 'adaptation'.")
    # clip-level or frame-level
    parser.add_argument("-level", '--level', type=str, default='frame',
                    help="Choose the level to do DA. 'clip' or 'frame'.")
    # Use fpn
    parser.add_argument("-fpn", '--use_fpn', action="store_true",
                    help="Whether to use CRNN_fpn architecture.")
    # Use Meanteacher
    parser.add_argument("-mt", '--meanteacher', action="store_true",
                    help="Whether to use mean teacher method.")
    # Use ICT, SCT, Weakly psuedo-labeling
    parser.add_argument("-ISP", '--ISP', action="store_true",
                    help="Whether to use three semi-supervised learning strategies.")


    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    stage = f_args.stage
    meanteacher = f_args.meanteacher
    ISP = f_args.ISP
    if ISP:
        meanteacher = True

    model_name = 'test_adaptation_FPN'# name your own model

    store_dir = os.path.join("stored_data", model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    start_epoch = 0
    if start_epoch == 0:
        writer = SummaryWriter(os.path.join(store_dir, "log"))
        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(saved_model_dir, exist_ok=True)
        os.makedirs(saved_pred_dir, exist_ok=True)
    else:
        writer = SummaryWriter(os.path.join(store_dir, "log"), purge_step=start_epoch)

    n_channel = 1
    add_axis_conv = 0

    # Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
    n_layers = 7
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
    
    discriminator_kwargs = {"input_dim": 256, "dropout": 0.5} # default 256
    predictor_kwargs = {"nclass":len(cfg.classes), "attention":True, "n_RNN_cell":128}

    pooling_time_ratio = 4  # 2 * 2

    out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    logger.debug(f"median_window: {median_window}")
    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)
    dfs = get_dfs(dataset, reduced_number_of_data)

    # Meta path for psds
    durations_synth = get_durations_df(cfg.synthetic)
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df

    # Normalisation per audio or on the full dataset
    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
        unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms)
        train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms)
        scaler_args = []
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data, train_synth_data]))
        logger.debug(f"scaler mean: {scaler.mean_}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)

    weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms, in_memory=cfg.in_memory_unlab)
    train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms, in_memory=cfg.in_memory)
    valid_synth_data = DataLoadDf(dfs["valid_synthetic"], encod_func, transforms_valid,
                                  return_indexes=True, in_memory=cfg.in_memory)

    # real validation data
    validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True, in_memory=cfg.in_memory)
    weak_dataload = DataLoadDf(dfs["validation"], many_hot_encoder.encode_weak, transforms_valid, return_indexes=True, in_memory=cfg.in_memory)
    logger.debug(f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}")

    if stage == 'adaptation' or (stage=='pretrain' and meanteacher):
        if not no_synthetic:
            list_dataset = [weak_data, unlabel_data, train_synth_data]
            batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
            strong_mask = slice((3*cfg.batch_size)//4, cfg.batch_size)
        else:
            list_dataset = [weak_data, unlabel_data]
            batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
            strong_mask = None
        weak_mask = slice(batch_sizes[0])  # Assume weak data is always the first one
    else:
        list_dataset = [weak_data, train_synth_data]
        batch_sizes = [cfg.batch_size//2, cfg.batch_size//2]
        strong_mask = slice((cfg.batch_size)//2, cfg.batch_size)
        weak_mask = slice(batch_sizes[0])

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)
    training_loader = DataLoader(concat_dataset, batch_sampler=sampler)
    valid_synth_loader = DataLoader(valid_synth_data, batch_size=cfg.batch_size)
    # real validation data
    validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    validation_dataloader_weak = DataLoader(weak_dataload, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ##############
    # Model
    # ##############
    if f_args.use_fpn:
        crnn = CRNN_fpn(**crnn_kwargs)
    else:
        crnn = CRNN(**crnn_kwargs)
    pytorch_total_params = sum(p.numel() for p in crnn.parameters() if p.requires_grad)
    logger.info(crnn)
    logger.info("number of parameters in the model: {}".format(pytorch_total_params))
    

    if stage == 'adaptation':
        if f_args.level == 'frame':
            discriminator = Frame_Discriminator(**discriminator_kwargs)
        elif f_args.level == 'clip':
            discriminator = Clip_Discriminator(**discriminator_kwargs)
    else:
        discriminator = None

    predictor = Predictor(**predictor_kwargs)
    
    if meanteacher:
        if f_args.use_fpn:
            crnn_ema = CRNN_fpn(**crnn_kwargs)
        else:
            crnn_ema = CRNN(**crnn_kwargs) 
        predictor_ema = Predictor(**predictor_kwargs)

    if start_epoch == 0:
        crnn.apply(weights_init)
        if stage == 'adaptation':
            discriminator.apply(weights_init)
        predictor.apply(weights_init)

        if meanteacher:
            crnn_ema.apply(weights_init)
            predictor_ema.apply(weights_init)
    # resume training
    else:        
        model_path = os.path.join(saved_model_dir, 'baseline_epoch_{}'.format(start_epoch-1))
        expe_state = torch.load(model_path, map_location="cpu")
        
        if not f_args.use_fpn:
            for key in list(expe_state["model"]["state_dict"].keys()): # match keys
                if 'cnn.' in key:
                    expe_state["model"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = expe_state["model"]["state_dict"][key]
                    del expe_state["model"]["state_dict"][key]
        crnn.load_state_dict(expe_state["model"]["state_dict"])
        
        if stage == 'adaptation':
            if start_epoch == 1 or start_epoch == 51:
                discriminator.apply(weights_init) # for adversarial training
            else:
                discriminator.load_state_dict(expe_state["model_d"]["state_dict"])
        predictor.load_state_dict(expe_state["model_p"]["state_dict"])
        
        if meanteacher:
            if not f_args.use_fpn:
                for key in list(expe_state["model_ema"]["state_dict"].keys()): # match keys of teacher model
                    if 'cnn.' in key:
                        expe_state["model_ema"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = expe_state["model_ema"]["state_dict"][key]
                        del expe_state["model_ema"]["state_dict"][key]
            crnn_ema.load_state_dict(expe_state["model_ema"]["state_dict"])
            predictor_ema.load_state_dict(expe_state["model_p_ema"]["state_dict"])
            
    if meanteacher:
        for param in crnn_ema.parameters():
            param.detach_()
        for param in predictor_ema.parameters():
            param.detach_()

    optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    optim_d_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    optim_crnn_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, list(crnn.parameters())+list(predictor.parameters())), **optim_kwargs)
    optim_crnn = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_crnn_kwargs)
    if stage == 'adaptation':
        optim_d = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), **optim_d_kwargs)
        
        if start_epoch > 1 and start_epoch != 51:
            optim.load_state_dict(expe_state['optimizer']['state_dict'])
            optim_d.load_state_dict(expe_state['optimizer_d']['state_dict'])
            optim_crnn.load_state_dict(expe_state['optimizer_crnn']['state_dict'])
            # deal with "RuntimeError: expected device cpu but got device cuda:0"
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            for state in optim_d.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            for state in optim_crnn.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    else:
        optim_d = None


    if stage == 'pretrain':
        state = {
            'model': {"name": crnn.__class__.__name__,
                    'args': '',
                    "kwargs": crnn_kwargs,
                    'state_dict': crnn.state_dict()},
            'model_p': {"name": predictor.__class__.__name__,
                        'args': '',
                        "kwargs": predictor_kwargs,
                        'state_dict': predictor.state_dict()},
            'optimizer': {"name": optim.__class__.__name__,
                        'args': '',
                        "kwargs": optim_kwargs,
                        'state_dict': optim.state_dict()},
            'optimizer_crnn': {"name": optim_crnn.__class__.__name__,
                    'args': '',
                    "kwargs": optim_kwargs,
                    'state_dict': optim_crnn.state_dict()},
            "pooling_time_ratio": pooling_time_ratio,
            "scaler": {
                "type": type(scaler).__name__,
                "args": scaler_args,
                "state_dict": scaler.state_dict()},
            "many_hot_encoder": many_hot_encoder.state_dict(),
            "median_window": median_window,
            "desed": dataset.state_dict()
        }
    else:
        state = {
            'model': {"name": crnn.__class__.__name__,
                    'args': '',
                    "kwargs": crnn_kwargs,
                    'state_dict': crnn.state_dict()},
            'model_d': {"name": discriminator.__class__.__name__,
                        'args': '',
                        "kwargs": discriminator_kwargs,
                        'state_dict': discriminator.state_dict()},
            'model_p': {"name": predictor.__class__.__name__,
                        'args': '',
                        "kwargs": predictor_kwargs,
                        'state_dict': predictor.state_dict()},
            'optimizer': {"name": optim.__class__.__name__,
                        'args': '',
                        "kwargs": optim_kwargs,
                        'state_dict': optim.state_dict()},
            'optimizer_d': {"name": optim_d.__class__.__name__,
                    'args': '',
                    "kwargs": optim_kwargs,
                    'state_dict': optim_d.state_dict()},
            'optimizer_crnn': {"name": optim_crnn.__class__.__name__,
                    'args': '',
                    "kwargs": optim_kwargs,
                    'state_dict': optim_crnn.state_dict()},
            "pooling_time_ratio": pooling_time_ratio,
            "scaler": {
                "type": type(scaler).__name__,
                "args": scaler_args,
                "state_dict": scaler.state_dict()},
            "many_hot_encoder": many_hot_encoder.state_dict(),
            "median_window": median_window,
            "desed": dataset.state_dict()
        }

    if meanteacher:
        state["model_ema"] = {"name": crnn_ema.__class__.__name__,
                          'args': '',
                          "kwargs": crnn_kwargs,
                          'state_dict': crnn_ema.state_dict()}
        state["model_p_ema"] = {"name": predictor_ema.__class__.__name__,
                        'args': '',
                        "kwargs": predictor_kwargs,
                        'state_dict': predictor_ema.state_dict()}

    save_best_cb = SaveBest("sup")
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    # ##############
    # Train
    # ##############
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(start_epoch, cfg.n_epoch):
        crnn.train()
        if stage == 'adaptation':
            discriminator.train()
        predictor.train()
        # crnn, crnn_ema, predictor = to_cuda_if_available(crnn, crnn_ema, predictor)
        if stage == 'adaptation':
            crnn, discriminator, predictor = to_cuda_if_available(crnn, discriminator, predictor)
        else:
            crnn, predictor = to_cuda_if_available(crnn, predictor)
        
        if meanteacher:
            crnn_ema.train()
            predictor_ema.train()
            crnn_ema, predictor_ema = to_cuda_if_available(crnn_ema, predictor_ema)

            loss_value = train(training_loader, crnn, optim, epoch, ema_model=crnn_ema, ema_predictor=predictor_ema,
                            mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=cfg.adjust_lr, predictor=predictor, discriminator=discriminator, optimizer_d=optim_d, optimizer_crnn=optim_crnn, ISP=ISP)            
        else:
            loss_value = train(training_loader, crnn, optim, epoch,
                            mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=cfg.adjust_lr, predictor=predictor, discriminator=discriminator, optimizer_d=optim_d, optimizer_crnn=optim_crnn, ISP=ISP)

        # Validation
        crnn.eval()
        predictor.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_loader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      median_window=median_window, save_predictions=None, predictor=predictor)
        # Validation with synthetic data (dropping feature_filename for psds)
        valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        valid_synth_f1, psds_m_f1 = compute_metrics(predictions, valid_synth, durations_synth)
        writer.add_scalar('Strong F1-score', valid_synth_f1, epoch)
        # Real validation data
        validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
        durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
        if f_args.use_fpn:     
            valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                            pooling_time_ratio, median_window=median_window, predictor=predictor, fpn=True)
        else:
            valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                            pooling_time_ratio, median_window=median_window, predictor=predictor)
        valid_real_f1, psds_real_f1 = compute_metrics(valid_predictions, validation_labels_df, durations_validation)
        writer.add_scalar('Real Validation F1-score', valid_real_f1, epoch)
        # Evaluate weak
        weak_metric = get_f_measure_by_class(crnn, len(cfg.classes), validation_dataloader_weak, predictor=predictor)
        writer.add_scalar("Weak F1-score macro averaged", np.mean(weak_metric), epoch)  

        # Update state
        state['model']['state_dict'] = crnn.state_dict()
        state['model_p']['state_dict'] = predictor.state_dict()
        state['optimizer']['state_dict'] = optim.state_dict()
        state['optimizer_crnn']['state_dict'] = optim_crnn.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_synth_f1
        state['valid_f1_psds'] = psds_m_f1
        if stage == 'adaptation':
            state['model_d']['state_dict'] = discriminator.state_dict()
            state['optimizer_d']['state_dict'] = optim_d.state_dict()
        if meanteacher:
            state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['model_p_ema']['state_dict'] = predictor_ema.state_dict()

        # Callbacks
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)
            results.loc[epoch, "global_valid"] = valid_synth_f1
        results.loc[epoch, "loss"] = loss_value.item()
        results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

        if cfg.early_stopping:
            if early_stopping_call.apply(valid_synth_f1):
                logger.warn("EARLY STOPPING")
                break


    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        crnn = _load_crnn(state)
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info("testing model of last epoch: {}".format(cfg.n_epoch))
    results_df = pd.DataFrame(results).to_csv(os.path.join(saved_pred_dir, "results.tsv"),
                                              sep="\t", index=False, float_format="%.4f")
    # ##############
    # Validation
    # ##############
    crnn.eval()
    predictor.eval()
    # transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

    # validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
    # validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    # validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    # durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
    # Preds with only one value
    valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                        pooling_time_ratio, median_window=median_window,
                                        save_predictions=predicitons_fname, predictor=predictor)
    compute_metrics(valid_predictions, validation_labels_df, durations_validation)