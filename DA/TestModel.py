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
from models.CRNN import CRNN, Predictor, CRNN_fpn
import config as cfg
import pdb
import collections
import os

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
    if not use_fpn:
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
    use_fpn = args.use_fpn
    use_predictor = args.use_predictor

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

    return model_pth, median_win, gt_audio_pth, groundtruth, meta_dur_df, use_fpn, use_predictor


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
    parser.add_argument("-fpn", '--use_fpn', action="store_true",
                    help="Whether to use CRNN_fpn architecture.")  
    # Use predictor
    parser.add_argument("-pd", '--use_predictor', action="store_true",
                    help="Whether to use label predictor.")
    # Dir to save embedding feature
    parser.add_argument("-sf", '--saved_feature_dir', type=str, default=None,
                        help="Path for the embedded features to be saved (if needed), Ex. embedded_feature/ADDA_with_synthetic_clipD_meanteacher_ISP, which kinds of data (weak/strong/synthetic) would be automatically detected")

    f_args = parser.parse_args()
    # Get variables from f_args
    model_path, median_window, gt_audio_dir, groundtruth, durations, use_fpn, use_predictor = get_variables(f_args)

    if f_args.groundtruth_tsv.split('/')[-1] == 'validation.tsv':
        data_type = 'strong'
    elif f_args.groundtruth_tsv.split('/')[-1]== 'weak.tsv':
        data_type = 'weak'
    elif f_args.groundtruth_tsv.split('/')[-1] == 'soundscapes.tsv':
        data_type = 'synth'

    saved_feature_dir = os.path.join(f_args.saved_feature_dir, data_type)
    if not os.path.exists(saved_feature_dir):
        os.makedirs(saved_feature_dir, exist_ok=True)
    

    # Model
    expe_state = torch.load(model_path, map_location="cpu")

    dataset = DESED(base_feature_dir=osp.join(cfg.workspace, "dataset", "features"), compute_log=False)

    gt_df_feat = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, nb_files=f_args.nb_files)
    params = _load_state_vars(expe_state, gt_df_feat, median_window, use_fpn, use_predictor)

    # Preds with only one value
    if use_fpn:
        single_predictions = get_predictions(params["model"], params["strong_dataloader"],
                                            params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                            median_window=params["median_window"],
                                            save_predictions=f_args.save_predictions_path,
                                            predictor=params["predictor"], fpn=True, saved_feature_dir=saved_feature_dir)
    else:
        single_predictions = get_predictions(params["model"], params["strong_dataloader"],
                                            params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                            median_window=params["median_window"],
                                            save_predictions=f_args.save_predictions_path,
                                            predictor=params["predictor"], saved_feature_dir=saved_feature_dir)
    compute_metrics(single_predictions, groundtruth, durations)
    
    # Evaluate audio tagging
    weak_metric = get_f_measure_by_class(params["model"], len(cfg.classes), params["weak_dataloader"], predictor=params["predictor"])
    print("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, params["many_hot_encoder"].labels)))
    print("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
    pdb.set_trace()
 