# -*- coding: utf-8 -*-
import argparse
import os.path as osp

import torch
# from psds_eval import PSDSEval
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data_utils.DataLoad import DataLoadDf
from data_utils.Desed import DESED
# from evaluation_measures import compute_sed_eval_metrics, psds_score, get_predictions
from evaluation_measures import psds_score, get_predictions, \
    compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
from utilities.utils import to_cuda_if_available, generate_tsv_wav_durations, meta_path_to_audio_dir
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms
from utilities.Logger import create_logger
from utilities.Scaler import Scaler, ScalerPerAudio
# from models.CRNN_FPN import CRNN_fpn
# from models.CRNN_DANN import CRNN
# from models.CRNN import CRNN
# from models.CRNN_localatten import CRNN
# from models.CRNN_ADDA import CRNN, Predictor
from models.CRNN_DANN_v3 import CRNN, Predictor, CRNN_fpn
import config as cfg
import pdb
import collections

logger = create_logger(__name__)
torch.manual_seed(2020)


def _load_crnn(state, model_name="model", use_fpn=False):
    crnn_args = state[model_name]["args"]
    crnn_kwargs = state[model_name]["kwargs"]
    if use_fpn:
        crnn = CRNN_fpn(*crnn_args, **crnn_kwargs)
    else:
        crnn = CRNN(*crnn_args, **crnn_kwargs)

    # if "ADDA" in model_path:
    for key in list(expe_state["model"]["state_dict"].keys()):
        if 'cnn.' in key:
            expe_state["model"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = expe_state["model"]["state_dict"][key]
            del expe_state["model"]["state_dict"][key]

    crnn.load_state_dict(state[model_name]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(crnn)
    return crnn


def _load_scaler(state):
    scaler_state = state["scaler"]
    type_sc = scaler_state["type"]
    if type_sc == "ScalerPerAudio":
        scaler = ScalerPerAudio(*scaler_state["args"])
    elif type_sc == "Scaler":
        scaler = Scaler()
    else:
        raise NotImplementedError("Not the right type of Scaler has been saved in state")
    scaler.load_state_dict(state["scaler"]["state_dict"])
    return scaler


def _load_state_vars(state, gtruth_df, median_win=None, use_fpn=False, use_predictor=False):
    pred_df = gtruth_df.copy()
    # Define dataloader
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    scaler = _load_scaler(state)
    crnn = _load_crnn(state, use_fpn=use_fpn)
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis=0)
    
    strong_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_strong_df, transforms_valid, return_indexes=True)
    strong_dataloader_ind = DataLoader(strong_dataload, batch_size=cfg.batch_size, drop_last=False, shuffle=False)

    # weak dataloader
    weak_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_weak, transforms_valid, return_indexes=True)
    weak_dataloader_ind = DataLoader(weak_dataload, batch_size=cfg.batch_size, drop_last=False, shuffle=False)

    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_win is None:
        median_win = state["median_window"]
    if use_predictor == False:
        return {
            "model": crnn,
            "strong_dataloader": strong_dataloader_ind,
            "weak_dataloader": weak_dataloader_ind,
            "pooling_time_ratio": pooling_time_ratio,
            "many_hot_encoder": many_hot_encoder,
            "median_window": median_win,
            "predictor": None
        }
    else:
        predictor_args = state["model_p"]["args"]
        predictor_kwargs = state["model_p"]["kwargs"]
        predictor = Predictor(**predictor_kwargs)
        predictor.load_state_dict(expe_state["model_p"]["state_dict"])
        predictor.eval()

        '''
        # load crnn weight init w/o splitting into feature extractor and classifier 
        crnn_init = torch.load('stored_data/MeanTeacher_with_synthetic_fcn_4/model/baseline_best', map_location="cpu")["model"]["state_dict"] #torch.load('crnn_init', map_location="cpu")
        extractor_init = []
        predictor_init = []
        for key, value in crnn_init.items():
            if 'dense' not in key:
                extractor_init.append((key, value))
            else:
                predictor_init.append((key, value))
        extractor_init = collections.OrderedDict(extractor_init)    
        predictor_init = collections.OrderedDict(predictor_init) 
        crnn.load_state_dict(extractor_init)
        predictor.load_state_dict(predictor_init)
        crnn.eval()
        crnn = to_cuda_if_available(crnn)
        predictor.eval()
        '''
        

        predictor = to_cuda_if_available(predictor)
        return {
            "model": crnn,
            "strong_dataloader": strong_dataloader_ind,
            "weak_dataloader": weak_dataloader_ind,
            "pooling_time_ratio": pooling_time_ratio,
            "many_hot_encoder": many_hot_encoder,
            "median_window": median_win,
            "predictor": predictor
        }


def get_variables(args):
    model_pth = args.model_path
    gt_fname, ext = osp.splitext(args.groundtruth_tsv)
    median_win = args.median_window
    meta_gt = args.meta_gt
    gt_audio_pth = args.groundtruth_audio_dir
    if args.use_fpn=='T':
        use_fpn = True
    elif args.use_fpn=='F':
        use_fpn = False

    if args.learned_post=='T':    
        learned_post = True
    elif args.learned_post=='F':
        learned_post = False

    if args.use_predictor=='T':    
        use_predictor = True
    elif args.use_predictor=='F':
        use_predictor = False

    if meta_gt is None:
        meta_gt = gt_fname + "_durations" + ext

    if gt_audio_pth is None:
        gt_audio_pth = meta_path_to_audio_dir(gt_fname)
        # Useful because of the data format
        if "validation" in gt_audio_pth:
            gt_audio_pth = osp.dirname(gt_audio_pth)

    groundtruth = pd.read_csv(args.groundtruth_tsv, sep="\t")
    if osp.exists(meta_gt):
        meta_dur_df = pd.read_csv(meta_gt, sep='\t')
        if len(meta_dur_df) == 0:
            meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)
    else:
        meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)

    return model_pth, median_win, gt_audio_pth, groundtruth, meta_dur_df, use_fpn, learned_post, use_predictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", '--model_path', type=str, required=True,
                        help="Path of the model to be evaluated")
    parser.add_argument("-g", '--groundtruth_tsv', type=str, required=True,
                        help="Path of the groundtruth tsv file")

    # Not required after that, but recommended to defined
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")

    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-ga", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")

    # Dev
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")

    # Use fpn
    parser.add_argument("-fpn", '--use_fpn', type=str, default='F',
                    help="Whether to use CRNN_fpn architecture, must be same as the saved model.'T' for True, 'F' for False.")
    # Use adaptive post processing
    parser.add_argument("-lp", '--learned_post', type=str, default='F',
                    help="Whether to use adaptive post processing.'T' for True, 'F' for False.")    
    # use predictor
    parser.add_argument("-pd", '--use_predictor', type=str, default='F',
                    help="Whether to use adaptive post processing.'T' for True, 'F' for False.")    


    f_args = parser.parse_args()
    # Get variables from f_args
    model_path, median_window, gt_audio_dir, groundtruth, durations, use_fpn, learned_post, use_predictor = get_variables(f_args)
    # Model
    expe_state = torch.load(model_path, map_location="cpu")
    
    # if f_args.use_predictor == 'T': # "DANN" in model_path:
    #     # match keys
    #     for key in list(expe_state["model"]["state_dict"].keys()):
    #         if 'cnn.' in key:
    #             expe_state["model"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = expe_state["model"]["state_dict"][key]
    #             del expe_state["model"]["state_dict"][key]
    # if f_args.use_predictor == 'T': # "DANN" in model_path:
    #     # match keys
    #     for key in list(expe_state["model"]["state_dict"].keys()):
    #         if 'cnn.' in key:
    #             expe_state["model"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = expe_state["model"]["state_dict"][key]
    #             del expe_state["model"]["state_dict"][key]
    #         else:
    #             expe_state["model"]["state_dict"]['cnn.'+key] = expe_state["model"]["state_dict"][key]
    #             del expe_state["model"]["state_dict"][key]

    dataset = DESED(base_feature_dir=osp.join(cfg.workspace, "dataset", "features"), compute_log=False)

    gt_df_feat = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, nb_files=f_args.nb_files)
    params = _load_state_vars(expe_state, gt_df_feat, median_window, use_fpn, use_predictor)

    # crnn_params = [item for item in params['model'].named_parameters()]
    # predictor_params = [item for item in params['predictor'].named_parameters()]

    # parameter_list = list(params["model"].named_parameters())
    # pdb.set_trace()
    # Preds with only one value
    if f_args.use_fpn=='T':
        single_predictions = get_predictions(params["model"], params["strong_dataloader"],
                                            params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                            median_window=params["median_window"],
                                            save_predictions=f_args.save_predictions_path,
                                            learned_post=learned_post, predictor=params["predictor"], fpn=True)
    else:
        single_predictions = get_predictions(params["model"], params["strong_dataloader"],
                                            params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                            median_window=params["median_window"],
                                            save_predictions=f_args.save_predictions_path,
                                            learned_post=learned_post, predictor=params["predictor"])
    compute_metrics(single_predictions, groundtruth, durations)
    pdb.set_trace()
    # Evaluate weak
    weak_metric = get_f_measure_by_class(params["model"], len(cfg.classes), params["weak_dataloader"], predictor=params["predictor"])
    print("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, params["many_hot_encoder"].labels)))
    print("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
    pdb.set_trace()
    
    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    if f_args.use_fpn=='T':
        pred_ss_thresh = get_predictions(params["model"], params["strong_dataloader"],
                                        params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                        thresholds=list_thresholds, median_window=params["median_window"],
                                        save_predictions=f_args.save_predictions_path,
                                        learned_post=learned_post, fpn=True)
    else:
        pred_ss_thresh = get_predictions(params["model"], params["strong_dataloader"],
                                        params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                        thresholds=list_thresholds, median_window=params["median_window"],
                                        save_predictions=f_args.save_predictions_path,
                                        learned_post=learned_post)
    # pdb.set_trace()
    # compute_psds_from_operating_points(pred_ss_thresh, groundtruth, durations)
    psds = compute_psds_from_operating_points(pred_ss_thresh, groundtruth, durations)
    # psds_score(psds, filename_roc_curves=osp.splitext(f_args.save_predictions_path)[0] + "_roc.png")
    fname_roc = None
    if f_args.save_predictions_path is not None:
        fname_roc = osp.splitext(f_args.save_predictions_path)[0] + "_roc.png"
    psds_score(psds, filename_roc_curves=fname_roc)    